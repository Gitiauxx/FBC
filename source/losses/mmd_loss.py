import torch

from source.losses.templates import TemplateLoss

class MMDLoss(TemplateLoss):
    """
    Compute a rate distortion loss based variational loss that combines L2 reconstruction loss
    and mutual information between X and Z
    """

    def __init__(self, fairness=0, beta=0):
        super().__init__()
        self.fairness = fairness
        self.beta = beta

    def forward(self, target, output, sensitive):
        """

        :param x: (B, input_dim)
        :param y: (B, input_dim)
        :param sensitive: (B, zdim)
        :return: rec_loss + self.fairness * I(Z, X)
        """
        rec_loss = torch.mean((target - output) ** 2)
        s1 = output[sensitive == 1, ...]
        s0 = output[sensitive == 0, ...]

        mmd = self.compute_rbf_kernel(s0, s0)
        mmd += self.compute_rbf_kernel(s1, s1)
        mmd -= 2 * self.compute_rbf_kernel(s0, s1)

        return rec_loss + self.fairness * mmd

    def compute_rbf_kernel(self, x, y):
        """
        Compute Gaussian-based kernel between x and y
        :param x:
        :param y:
        :return:
        """
        xx = x[:, None, :]
        yy = y[None, ...]
        x_y = torch.sum((xx - yy) ** 2, -1)
        return torch.exp(- x_y / self.beta).mean()