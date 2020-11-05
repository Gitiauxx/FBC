"""
Util modules to build neural networks in pytorch
"""

import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Basic encoder block with 2d conv, batch norm and PReLU
    """

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=1),
            nn.BatchNorm2d(channel_out),
            nn.PReLU())

    def forward(self, x):
        """

        :param x: B, Cin, Win, Hin
        :return: B, Cout, Wout, Hout
        """
        if len(x.shape) < 4:
            x = x[:, None, ...]
        return self.block(x)


class DecoderBlock(nn.Module):
    "Basic Decoder with 2d conv transpose layers, batch norm and PReLU"

    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=1):
        super().__init__()

        self.dblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out,
                               kernel_size=kernel_size, padding=padding, stride=stride, dilation=1),
            nn.BatchNorm2d(channel_out),
            nn.PReLU()
        )

    def forward(self, x):
        """

        :param x: (B, Cin, Win, Wout)
        :return: (B, Cout, Win * stride
        """

        output = self.dblock(x)

        return output


class OneLayerFC(nn.Module):
    """Basic one hidden layer fully connected structure"""

    def __init__(self, in_dim, out_dim, activation='PRELU', p=0.0):
        super().__init__()

        if activation == 'RELU':
            self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        if activation == 'ELU':
            self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ELU())
        if activation == 'PReLU':
            self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.PReLU())
        if activation == 'sigmoid':
            self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        if activation == 'tanh':
            self.fc = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())

        self.p = p

    def forward(self, x):
        """

        :param x: (B, in_dim)
        :return: (B, out_dim)
        """
        
        output = self.fc(x)
        return output


class ResidualBlock(nn.Module):
    """
    Implement a residual block CNN with 2 convolution layers and one CNN-skip connection layer
    """

    def __init__(self, channels):
        super().__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=channels[0],
                                              out_channels=channels[1],
                                              kernel_size=(3, 3),
                                              stride=(2, 2),
                                              padding=1),
                                    nn.BatchNorm2d(channels[1]),
                                    nn.PReLU)

        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=channels[1],
                                              out_channels=channels[2],
                                              kernel_size=(1, 1),
                                              stride=(1, 1),
                                              padding=0),
                                    )

        self.conv_shortcut_1 = nn.Sequential(nn.Conv2d(in_channels=channels[0],
                                                       out_channels=channels[2],
                                                       kernel_size=(1, 1),
                                                       stride=(2, 2),
                                                       padding=0),
                                             nn.BatchNorm2d(channels[2]))

    def forward(self, x):
        """

        :param x: (B, W, H)
        :return: (B, out_dim)
        """
        shortcut = x

        out = self.conv_1(x)
        out = self.conv_2(out)

        shortcut = self.conv_shortcut_1(shortcut)

        out += shortcut
        out = nn.functional.relu(out)

        return out
