import torch
from torch.nn.modules import MSELoss

from source.losses.templates import TemplateLoss


class VAELoss(TemplateLoss):
    """
    Compute L2 reconstruction loss
    """

    def __init__(self, beta=0, gamma=0, sigma=0.5):
        super().__init__()
        self.name = 'VAELoss'
        self.beta = beta
        self.gamma = gamma
        self.sigma= sigma

    def forward(self, target, output, b, z, sensitive, eps=10**(-8)):
        """

        :param output: reconstructed input (B, C, W, H)
        :param target: initial input (B, C, W, H)
        :return: mean squared loss
        """
        z_mean, z_logvar = z[0], z[1]
        if len(target.shape) == 2:
            N = target.shape[-1]
        elif len(target.shape) == 3:
            N = target.shape[-1] * target.shape[-2]

        KL = 1 / 2 * torch.mean(torch.sum(z_mean ** 2 - z_logvar + torch.exp(z_logvar) - 1, -1))
        rec_loss = (target - output) ** 2
        rec_loss = torch.mean(rec_loss.reshape(target.shape[0], -1).sum(-1))

        if len(sensitive.shape) == 1:
            s1 = b[sensitive == 1, ...]
            s0 = b[sensitive == 0, ...]

            mmd_loss = self.compute_rbf_kernel(s0, s0, reduction='mean')
            mmd_loss += self.compute_rbf_kernel(s1, s1, reduction='mean')
            mmd_loss -= 2 * self.compute_rbf_kernel(s0, s1, reduction='mean')

        elif len(sensitive.shape) == 2:
            pi_s = sensitive.sum(0)
            sx = sensitive[:, None, ...]
            sy = sensitive[None, ...]

            mmd = self.compute_rbf_kernel(b, b)
            mmd = mmd[..., None]
            mmd_x = torch.sum(mmd * sx * sy, dim=(0, 1)) / (pi_s ** 2 + eps)
            mmd_cross = torch.sum(mmd * sx, dim=(0,1)) / (eps + pi_s * sensitive.shape[0])
            mmd_all = torch.sum(mmd, dim=(0,1)) / (sensitive.shape[0]) ** 2

            mmd_loss = mmd_x + mmd_all - 2 * mmd_cross
            mmd_loss = torch.sum(mmd_loss)

        return self.beta * KL + rec_loss + self.gamma * mmd_loss

    def compute_rbf_kernel(self, x, y, reduction=None):
        """
        Compute Gaussian-based kernel between x and y
        :param x:
        :param y:
        :return:
        """
        xx = x[:, None, :]
        yy = y[None, ...]
        x_y = torch.sum((xx - yy) ** 2, -1)
        if reduction == 'mean':
            return torch.exp(- x_y / self.sigma).mean()
        else:
            return torch.exp(- x_y / self.sigma)
