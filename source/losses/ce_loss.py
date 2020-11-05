import torch
import torch.nn as nn

from source.losses.templates import TemplateLoss


class CECondLoss(TemplateLoss):
    """
    Implement a cross entropy loss with logits as torch BCEWithLogitsLoss
    """

    def __init__(self):
        super().__init__()

    def forward(self, target, prelogits):
        """

        :param target: (B, zdim, k)
        :param prelogits: (B, zdim, k)
        :return: (zdim, k) - target * log(sigmoid(prelogits)) - (1 - target) * log(1 - sigmoid(prelogits))
        """

        loss = nn.BCEWithLogitsLoss(reduction='none').forward(prelogits, target)

        # loss_total = 0
        # for i in range(2, prelogits.shape[1]):
        #     loss_total += loss[:, :i, :].view(target.shape[0], -1).sum(-1).mean()
        #
        # return loss_total

        return loss.view(target.shape[0], -1).sum(-1).mean()
