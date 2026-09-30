"""VAE and conditional diffusion model of the generative framework (paper Fig. 1a, Eqs. 10-15).

Architectures follow the released ``Function.py`` of the official repository:

* ``PhysicsVAE`` = ``SigmaVAE`` (the variant loaded by the notebooks): encoder Conv(3->32, 3x3, s1) -> Conv(32->64,
  4x4, s2) -> Conv(64->128, 5x5, s2) -> flatten -> two linear heads (mu, log-variance) of a 128-D latent; decoder
  linear -> ConvT(128->64, 6x6, s2) -> ConvT(64->32, 6x6, s2) -> ConvT(32->3, 5x5, s1) -> sigmoid.
  (three convolutional + three transposed-convolutional layers, as stated in the Methods.)
* ``Diffusion`` = DDIM-style conditional denoiser: continuous "offset cosine" noise schedule, the noisy image, a
  sinusoidal embedding of the noise variance (3 frequencies -> 6 channels) and the VAE output are concatenated
  (3 + 6 + 3 = 12 channels) and fed to a residual U-Net (32-64-96 channels, 128 in the middle) that predicts the
  noise.  Parameter names match the official ``diffusVAE.pt`` so that checkpoint can be loaded directly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import kl_divergence, physics_loss


class PhysicsVAE(nn.Module):
    def __init__(self, img_size=128, z_dim=128, filters=32, channels=3):
        super().__init__()
        f = filters
        self.z_dim, self.img_size = z_dim, img_size
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, f, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(f, 2 * f, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(2 * f, 4 * f, 5, 2, 2), nn.ReLU(),
            nn.Flatten())
        self.feat = (4 * f, img_size // 4, img_size // 4)
        h_dim = 4 * f * (img_size // 4) ** 2
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * f, 2 * f, 6, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(2 * f, f, 6, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(f, channels, 5, 1, 2), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(self.fc2(z).view(-1, *self.feat))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def loss(self, x_in, x_target, terms=('pix', 'edge', 'ssim'), kl_weight=1.0):
        """Eq. (11): L_pix + L_edge + L_SSIM + L_KL.  The KL term (summed over the 128 latent dimensions) is
        divided by the number of image elements so that it is on the same per-element scale as the
        reconstruction terms, i.e. a per-element ELBO; ``kl_weight`` rescales it further."""
        x_hat, mu, logvar = self(x_in)
        rec, parts = physics_loss(x_hat, x_target, terms)
        kl = kl_divergence(mu, logvar)
        parts['kl'] = kl / x_target[0].numel()
        return rec + kl_weight * parts['kl'], parts, x_hat


# ----------------------------------------------------------------------------------------------- diffusion model
def offset_cosine_schedule(t, min_signal=0.02, max_signal=0.95):
    """t in [0, 1] -> (noise_rate, signal_rate) with signal_rate^2 + noise_rate^2 = 1 (official 'offset_cosine')."""
    start, end = math.acos(max_signal), math.acos(min_signal)
    angle = start + t * (end - start)
    return torch.sin(angle), torch.cos(angle)


class Residual(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cnn = nn.Conv2d(cin, cout, 1)
        self.s = nn.Sequential(nn.BatchNorm2d(cin), nn.Conv2d(cin, cout, 3, 1, 1), nn.SiLU(),
                               nn.Conv2d(cout, cout, 3, 1, 1))

    def forward(self, x):
        return self.cnn(x) + self.s(x)


class UNet(nn.Module):
    def __init__(self, cin=12, cout=3):
        super().__init__()
        self.down = nn.ModuleList([Residual(cin, 32), Residual(32, 32), nn.AvgPool2d(2),
                                   Residual(32, 64), Residual(64, 64), nn.AvgPool2d(2),
                                   Residual(64, 96), Residual(96, 96), nn.AvgPool2d(2)])
        self.middle = nn.ModuleList([Residual(96, 128), Residual(128, 128)])
        self.up = nn.ModuleList([nn.UpsamplingBilinear2d(scale_factor=2), Residual(224, 96), Residual(192, 96),
                                 nn.UpsamplingBilinear2d(scale_factor=2), Residual(160, 64), Residual(128, 64),
                                 nn.UpsamplingBilinear2d(scale_factor=2), Residual(96, 32), Residual(64, 32)])
        self.out = nn.Conv2d(32, cout, 1)

    def forward(self, x):
        skips = []
        for m in self.down:
            x = m(x)
            if isinstance(m, Residual):
                skips.append(x)
        for m in self.middle:
            x = m(x)
        for m in self.up:
            if isinstance(m, Residual):
                x = torch.cat([x, skips.pop()], 1)
            x = m(x)
        return self.out(x)


class Combine(nn.Module):
    """Concatenate noisy image, sinusoidal noise-variance embedding and the VAE condition."""

    def __init__(self):
        super().__init__()
        self.register_buffer('t', torch.linspace(0.0, math.log(1000.0), 3).exp() * 2 * math.pi)
        # present (unused) in the official checkpoint; kept so that its state_dict loads strictly
        self.cnn_img = nn.Conv2d(3, 32, 1)
        self.cnn_vae = nn.Conv2d(3, 32, 1)

    def forward(self, image, var, cond):
        v = self.t * var.flatten(1)                                             # (b, 3)
        v = torch.cat([v.sin(), v.cos()], 1)[:, :, None, None].expand(-1, -1, *image.shape[-2:])
        return torch.cat([image, v, cond], 1)


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3, affine=False)    # identity statistics in the official checkpoint
        self.unet = UNet()
        self.combine = Combine()

    def predict_noise(self, x_t, noise_rate, cond):
        return self.unet(self.combine(x_t, noise_rate ** 2, cond))

    def loss(self, x0, cond, terms=('pix', 'ssim', 'edge')):
        """Eq. (15): physics-aware loss between predicted and true noise at a random diffusion time."""
        b = x0.shape[0]
        noise = torch.randn_like(x0)
        noise_rate, signal_rate = offset_cosine_schedule(torch.rand(b, 1, 1, 1, device=x0.device))
        x_t = signal_rate * x0 + noise_rate * noise
        pred = self.predict_noise(x_t, noise_rate, cond)
        total, parts = physics_loss(pred, noise, terms)
        return total, parts

    @torch.no_grad()
    def sample(self, cond, steps=20, generator=None):
        """Deterministic DDIM sampling (official ``generate``): start from noise, 20 steps, clip to [0, 1]."""
        b = cond.shape[0]
        x = torch.randn(cond.shape, device=cond.device, generator=generator)
        mean = self.norm.running_mean.view(1, 3, 1, 1)
        std = self.norm.running_var.sqrt().view(1, 3, 1, 1)
        pred_x0 = x
        for i in range(steps):
            t = torch.full((b, 1, 1, 1), (steps - i) / steps, device=cond.device)
            noise_rate, signal_rate = offset_cosine_schedule(t)
            pred_noise = self.predict_noise(x, noise_rate, cond)
            pred_x0 = (x - noise_rate * pred_noise) / signal_rate
            noise_rate, signal_rate = offset_cosine_schedule(t - 1.0 / steps)
            x = signal_rate * pred_x0 + noise_rate * pred_noise
        return (mean + pred_x0 * std).clamp(0.0, 1.0)
