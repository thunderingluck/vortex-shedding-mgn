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
        recon = (h_hat - h).pow(2).sum(-1).mean()
        l0 = (z > 0).float().sum(-1).mean()
        return recon, recon.detach(), l0.detach()
