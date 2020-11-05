import torch.nn as nn

from source.losses.templates import TemplateLoss


class CELoss(TemplateLoss):
    """
    Implement a cross entropy loss with logits as torch BCEWithLogitsLoss
    """

    def __init__(self):
        super().__init__()

    def forward(self, target, prelogits):
        """

        :param target: (B)
        :param prelogits: (B, hdim)
        :return: - target * log(sigmoid(prelogits)) - (1 - target) * log(1 - sigmoid(prelogits))
        """
        target = target[:, None]

        # weights = target.shape[0] * (target / target.sum() + (1 - target ) / (1 -target).sum())
        return nn.BCEWithLogitsLoss().forward(prelogits, target.float())


class MCELoss(TemplateLoss):

    def __init__(self):
        super().__init__()

    def forward(self, target, prelogits):
        """
        :param target: (B)
        :param prelogits: (B, hdim)
        :return: - E log(softmax(prelogits))
        """

        return nn.CrossEntropyLoss().forward(prelogits, target.long())
