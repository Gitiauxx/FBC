import torch.nn as nn

from source.model_utils import OneLayerFC, EncoderBlock
from source.template_model import TemplateModel


class CNNAud(TemplateModel):
    """
    Implement a basic fully connected mlp with RELU/PRELU/Sigmoid/tanh activation
    to predict sensitive attribute. Returns logit or activation_out(logit)
    """

    def __init__(self, num_enc_layer, ratio, in_channels=3,
                 kernel_size=4, out_channels=16, stride=2, nclass=1, activation_out=None):
        super().__init__()

        mul = 1
        input_channel = in_channels

        hidden_blocks = []
        for _ in range(num_enc_layer):
            hidden_blocks.append(EncoderBlock(in_channels, out_channels * mul, kernel_size=kernel_size, stride=stride))
            in_channels = mul * out_channels
            mul *= 2

        self.maps_dim = in_channels

        self.hidden_layers = nn.Sequential(*hidden_blocks)
        self.final_layer = nn.Linear(self.maps_dim * ratio * ratio, nclass)

        if activation_out is not None:
            self.final_layer = OneLayerFC(self.maps_dim * ratio * ratio, nclass, activation=activation_out)

        self.param_init()

    def forward(self, x):
        """

        :param x: (B, zdim)
        :return: (B, 1)
        """
        output = self.hidden_layers(x)
        out = output.view(x.shape[0], -1)
        output = self.final_layer(out)

        return output

    def param_init(self):
        """
        Xavier's initialization
        """
        for layer in self.modules():
            if hasattr(layer, 'weight'):

                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.PReLU, nn.Tanh)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)
