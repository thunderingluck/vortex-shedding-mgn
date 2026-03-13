import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    SAE matching the paper:
      z = ReLU((h - b_dec) W_enc + b_enc)
      h_hat = z W_dec + b_dec
    with decoder row renormalization to unit L2 after init and each step.
    Loss: ||h_hat - h||_2^2 + lam * ||z||_1
    :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, d_in: int, expansion: int = 8):
        super().__init__()
        d_hid = expansion * d_in
        self.d_in = d_in
        self.d_hid = d_hid

        # Note: paper uses (h - b_dec) in encoder, so b_dec participates there too.
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_in, d_hid))
        self.b_enc = nn.Parameter(torch.zeros(d_hid))

        self.W_dec = nn.Parameter(torch.empty(d_hid, d_in))

        self.reset_parameters()
        self.renorm_decoder_rows_()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)
        # Initialize b_enc negative so most features are off by default.
        # Magnitude matches the expected pre-activation scale: sqrt(2/d_in).
        with torch.no_grad():
            self.b_enc.fill_(-((2 / self.d_in) ** 0.5))

    @torch.no_grad()
    def renorm_decoder_rows_(self, eps: float = 1e-8):
        # Normalize each row (dictionary atom) to unit L2 norm
        # Paper: "rows of Wdec are re-normalised to unit l2 norm after init and after every optimization step"
        # :contentReference[oaicite:3]{index=3}
        norms = torch.linalg.norm(self.W_dec, dim=1, keepdim=True).clamp_min(eps)
        self.W_dec.div_(norms)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        # h: (..., d_in)
        return F.relu((h - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, h: torch.Tensor):
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z

    def loss(self, h: torch.Tensor, lam: float):
        h_hat, z = self.forward(h)
        # Paper Eq. (6): ||h_hat - h||_2^2 + λ||z||_1
        # Both are per-sample norms (sum over dims), then averaged over the batch.
        recon = (h_hat - h).pow(2).sum(-1).mean()
        spars = z.abs().sum(-1).mean()
        return recon + lam * spars, recon.detach(), spars.detach()