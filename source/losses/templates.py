from torch.nn.modules import MSELoss


class TemplateLoss(MSELoss):
    """
    Template class to construct loss from config file
    """

    def __init__(self):
        super().__init__()
        self.name = 'template'

    @classmethod
    def from_dict(cls, config):
        """
        create an input configuration from a dictionary

        :param: config: configuration dictionaary
        :return: an architecture
        """
        return cls(**config)
