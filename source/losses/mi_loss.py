import torch

from source.losses.templates import TemplateLoss

class MILoss(TemplateLoss):
    """
    Compute a rate distortion loss based variational loss that combines L2 reconstruction loss
    and mutual information between X and Z
    """

    def __init__(self, gamma=0):
        super().__init__()
        self.gamma = gamma

    def forward(self, target, output, eps = 10**(-8)):
        """

        :param x: (B, input_dim)
        :param y: (B, input_dim)
        :param mask: (B, zdim)
        :return: rec_loss + beta * bit_rate
        """

        rec_loss = (target - output) ** 2
        rec_loss = torch.mean(rec_loss.reshape(target.shape[0], -1).sum(-1))
        #s_loss = torch.nn.CrossEntropyLoss().forward(b, sensitive[..., 0].argmax(-1).long())

        # if len(sensitive.shape) > 1:
        #
        #     if len(b.shape) == 3:
        #         sensitive = sensitive[:, None, None, ...]
        #
        #     b = b[..., None]
        #     pi_s = torch.sum(sensitive, 0)
        #
        #     b0 = torch.sum(b * sensitive, 0) / (pi_s + eps)
        #     b1 = torch.sum(b * (1 - sensitive), 0) /(sensitive.shape[0] - pi_s + eps)
        #
        # else:
        #     b0 = torch.mean(b[sensitive == 0, ...], 0)
        #     b1 = torch.mean(b[sensitive == 1, ...], 0)

        return rec_loss
               #+ self.gamma * s_loss
