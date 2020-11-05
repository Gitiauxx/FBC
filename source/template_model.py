import torch.nn as nn


class TemplateModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.name = 'template'

    @classmethod
    def from_dict(cls, config):
        """
        create an input configuration from a dicitonary

        :param: config: configuration dictionaary
        :return: an architecture
        """
        return cls(**config)
