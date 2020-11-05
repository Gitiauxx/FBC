import torch
import torch.nn as nn
from source.template_model import TemplateModel


class MaskedCNN(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al.
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"

        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            self.mask[:, :, height // 2, width // 2:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0
        else:
            self.mask[:, :, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, height // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)

class MaskedCNN3D(nn.Conv3d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al.
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"

        self.register_buffer('mask', self.weight.data)

        _, _, depth, height, width = self.weight.size()
        self.mask.fill_(1)

        if mask_type == 'A':
            self.mask[:, :, depth // 2, height // 2, width // 2:] = 0
            self.mask[:, :, depth // 2, height // 2 + 1:, :] = 0
            self.mask[:, :, depth // 2 + 1:, :, :] = 0
        else:
            self.mask[:, :, depth //2 + 1:, height // 2, width // 2 + 1:] = 0
            self.mask[:, :, depth // 2+ 1:, height // 2 + 1:, :] = 0
            self.mask[:, :, depth // 2 + 1:, :, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(TemplateModel):

    def __init__(self, kernel=7, channels=24, in_features=(10, 10)):
        super().__init__()

        self.conv1 = MaskedCNN('A', 1, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
        self.ReLU_1 = nn.ReLU(True)

        self.conv2 = MaskedCNN('B', channels, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm2d_2 = nn.BatchNorm2d(channels)
        self.ReLU_2 = nn.ReLU(True)

        self.conv3 = MaskedCNN('B', channels, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm2d_3 = nn.BatchNorm2d(channels)
        self.ReLU_3 = nn.ReLU(True)

        self.conv4 = MaskedCNN('B', channels, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm2d_4 = nn.BatchNorm2d(channels)
        self.ReLU_4 = nn.ReLU(True)

        self.conv5 = MaskedCNN('B', channels, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm2d_5 = nn.BatchNorm2d(channels)
        self.ReLU_5 = nn.ReLU(True)

        self.out = nn.Conv2d(channels, 1, 1, 1)

    def forward(self, xin, s, use_s=False):
        """

        :param x: (B, zdim, k)
        :return: (B, zdim, k)
        """
        if len(xin.shape) == 3:
            x = xin[:, None, ...]
        elif len(xin.shape) == 2:
            x = xin[None, None, ...]
        else:
            x = xin


        s = s[:, None, None, None]

        x = self.conv1(x)
        x = self.BatchNorm2d_1(x)
        x = self.ReLU_1(x)

        x = self.conv2(x)
        x = self.BatchNorm2d_2(x)
        x = self.ReLU_2(x)

        x = self.conv3(x)
        x = self.BatchNorm2d_3(x)
        x = self.ReLU_3(x)

        x = self.conv4(x)
        x = self.BatchNorm2d_4(x)
        x = self.ReLU_4(x)

        x = self.conv5(x)
        x = self.BatchNorm2d_5(x)
        x = self.ReLU_5(x)

        x = self.out(x)

        return x[:, 0, ...]


class PixelCNN3D(TemplateModel):

    def __init__(self, kernel=7, channels=24, out_channels=128):
        super().__init__()

        self.conv1 = MaskedCNN3D('A', 1, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm3d_1 = nn.BatchNorm3d(channels)
        self.ReLU_1 = nn.ReLU(True)

        self.conv2 = MaskedCNN3D('B', channels, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm3d_2 = nn.BatchNorm3d(channels)
        self.ReLU_2 = nn.ReLU(True)

        self.conv3 = MaskedCNN3D('B', channels, channels, kernel, 1, kernel // 2, bias=False)
        self.BatchNorm3d_3 = nn.BatchNorm3d(channels)
        self.ReLU_3 = nn.ReLU(True)

        self.out = nn.Conv3d(channels, out_channels, 1)

    def forward(self, x):
        """

        :param x: (B, C, W, H)
        :return: (B, C, W, H)
        """
        x = x[:, None, ...]
        x = self.conv1(x)
        x = self.BatchNorm3d_1(x)
        x = self.ReLU_1(x)
        x = x.detach()

        x = self.conv2(x)
        x = self.BatchNorm3d_2(x)
        x = self.ReLU_2(x)
        x = x.detach()

        x = self.conv3(x)
        x = self.BatchNorm3d_3(x)
        x = self.ReLU_3(x)
        x = x.detach()

        x = self.out(x)

        return x[:, :, 0, ...]
