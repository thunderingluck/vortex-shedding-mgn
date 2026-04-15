import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoderTopK(nn.Module):
    """
    Top-K SAE: identical to SparseAutoencoder except the activation function
    is a hard top-K instead of ReLU + L1 penalty.

      pre   = (h - b_dec) @ W_enc + b_enc
      z     = top-K of pre, with ReLU applied to the selected values
      h_hat = z @ W_dec + b_dec

    Loss: reconstruction MSE only — no L1, no lambda.
    Sparsity is enforced by architecture: exactly K features fire per sample
    (fewer if fewer than K pre-activations are positive, rare after warm-up).

    Decoder rows are renormalised to unit L2 norm after init and each step,
    matching the original SAE convention.
    """

    def __init__(self, d_in: int, expansion: int = 8, k: int = 32):
        super().__init__()
        d_hid = expansion * d_in
        self.d_in = d_in
        self.d_hid = d_hid
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_hid))
        self.b_enc = nn.Parameter(torch.zeros(d_hid))
        self.W_dec = nn.Parameter(torch.empty(d_hid, d_in))

        self.reset_parameters()
        self.renorm_decoder_rows_()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)
        with torch.no_grad():
            self.b_enc.fill_(-((2 / self.d_in) ** 0.5))

    @torch.no_grad()
    def renorm_decoder_rows_(self, eps: float = 1e-8):
        norms = torch.linalg.norm(self.W_dec, dim=1, keepdim=True).clamp_min(eps)
        self.W_dec.div_(norms)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., d_in)
        pre = (h - self.b_dec) @ self.W_enc + self.b_enc  # (..., d_hid)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, h: torch.Tensor):
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z

    def loss(self, h: torch.Tensor):
        h_hat, z = self.forward(h)
        recon = (h_hat - h).pow(2).mean()
        l0 = (z > 0).float().sum(-1).mean()
        return recon, recon.detach(), l0.detach()

    def loss_with_aux(
        self,
        h: torch.Tensor,
        dead_mask: torch.Tensor,
        k_aux: int | None = None,
        alpha: float = 1 / 32,
    ):
        """MSE loss + auxiliary loss that gives gradient to dead features.

        Dead features are identified by the caller (via firing-frequency EMA).
        For each sample we find the top-k_aux dead features, reconstruct h from
        them alone, and add a scaled MSE of that reconstruction.  This prevents
        dead features from being permanently cut off from gradient signal.

        Args:
            dead_mask: bool tensor of shape (d_hid,), True where feature is dead.
            k_aux:     how many dead features to activate per sample (default: k).
            alpha:     weight of the auxiliary loss term.

        Returns:
            total_loss, recon_detached, l0_detached, aux_loss_detached
        """
        h_hat, z = self.forward(h)
        recon = (h_hat - h).pow(2).mean()
        l0 = (z > 0).float().sum(-1).mean()

        aux_loss = recon.new_tensor(0.0)
        if dead_mask.any():
            k_aux = k_aux if k_aux is not None else self.k
            pre = (h - self.b_dec) @ self.W_enc + self.b_enc  # (batch, d_hid)
            dead_pre = pre[:, dead_mask]                        # (batch, n_dead)
            k_eff = min(k_aux, dead_mask.sum().item())
            topk_vals, topk_idx = torch.topk(dead_pre, k_eff, dim=-1)
            dead_z = torch.zeros_like(dead_pre)
            dead_z.scatter_(-1, topk_idx, F.relu(topk_vals))
            # reconstruct using only dead-feature decoder columns
            aux_h_hat = dead_z @ self.W_dec[dead_mask] + self.b_dec
            aux_loss = (aux_h_hat - h).pow(2).mean()

        total = recon + alpha * aux_loss
        return total, recon.detach(), l0.detach(), aux_loss.detach()
