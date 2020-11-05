import torch

from source.losses.templates import TemplateLoss


class L2Loss(TemplateLoss):
    """
    Compute L2 reconstruction loss
    """

    def __init__(self, beta=0):
        super().__init__()
        self.name = 'L2Loss'
        self.beta = beta

    def forward(self, target, output):
        """

        :param output: reconstructed input (B, C, W, H)
        :param target: initial input (B, C, W, H)
        :return: mean squared loss
        """
        rec_loss = (target - output) ** 2
        rec_loss = torch.mean(rec_loss.reshape(target.shape[0], -1).sum(-1))

        return rec_loss
