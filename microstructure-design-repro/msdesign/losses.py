"""Physics-aware reconstruction losses (paper Eqs. 1-9, 11, 15).

L_pix  : mean absolute error                                  -> global intensity (Euler angles)
L_edge : L1 distance of Sobel responses in x and y            -> grain boundaries / interfaces
L_SSIM : 1 - SSIM (Gaussian 11x11 window, sigma 1.5, a=b=g=1) -> intra/intergranular orientation structure
"""
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.t().contiguous()


def pixel_loss(x_hat, x):
    return (x_hat - x).abs().mean()


def sobel(x):
    c = x.shape[1]
    kx = SOBEL_X.to(x).expand(c, 1, 3, 3)
    ky = SOBEL_Y.to(x).expand(c, 1, 3, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode='replicate')
    return F.conv2d(xp, kx, groups=c), F.conv2d(xp, ky, groups=c)


def edge_loss(x_hat, x):
    gx_h, gy_h = sobel(x_hat)
    gx, gy = sobel(x)
    return ((gx - gx_h).abs() + (gy - gy_h).abs()).mean()


def _gaussian_window(size=11, sigma=1.5, device=None, dtype=None):
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(x_hat, x, data_range=1.0, window=11, sigma=1.5, reduction='mean'):
    """SSIM (Wang et al. 2004), Eq. (8), computed per channel with a Gaussian window."""
    c = x.shape[1]
    w = _gaussian_window(window, sigma, x.device, x.dtype).expand(c, 1, window, window)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(x_hat, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(x_hat * x_hat, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * x_hat, w, groups=c) - mu_x * mu_y
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return s.mean() if reduction == 'mean' else s.flatten(1).mean(1)


def ssim_loss(x_hat, x, data_range=1.0):
    return 1.0 - ssim(x_hat, x, data_range=data_range)


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)) summed over latent dimensions, averaged over the batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()


def physics_loss(x_hat, x, terms=('pix', 'edge', 'ssim'), data_range=1.0):
    """Sum of the selected reconstruction terms; returns (total, dict of components)."""
    parts = {}
    if 'pix' in terms:
        parts['pix'] = pixel_loss(x_hat, x)
    if 'edge' in terms:
        parts['edge'] = edge_loss(x_hat, x)
    if 'ssim' in terms:
        parts['ssim'] = ssim_loss(x_hat, x, data_range)
    return sum(parts.values()), parts
