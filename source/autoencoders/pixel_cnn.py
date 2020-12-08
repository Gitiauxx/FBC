import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from source.template_model import TemplateModel

class AutoReg(nn.Module):

    def __init__(self, in_features, out_features, k, mask_type='A'):
        super().__init__()

        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        self.bias = Parameter(torch.FloatTensor(out_features), requires_grad=True)

        assert mask_type in ['A', 'B'], "Unknown Mask Type"

        if mask_type == 'A':
            self.register_buffer('mask', torch.ones(k, in_features, out_features))
            for i in range(k - 1):
                self.mask[i, (i+1):, :] = 0

        if mask_type == 'B':
            self.register_buffer('mask', torch.ones(k, in_features, out_features))

        self.mask_type = mask_type
        self.param_init()

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


    def forward(self, x):
        if self.mask_type == 'A':
            w = self.weight[None, ...] * self.mask
            xw = torch.matmul(x, w)
            xw = xw.permute(1, 2, 0)

        else:
            xx = x.transpose(-2, -1)
            xw = torch.matmul(xx, self.weight)
            xw = xw.transpose(-2, -1)

        return xw + self.bias[None, :, None]


class RegLayer(TemplateModel):

    def __init__(self, in_features, out_dim, out_features=64, nclass=0 , k=10, activation_out=None):
        super().__init__()

        self.layer1 = AutoReg(in_features, out_features, k, mask_type='A')
        self.ReLU_1 = nn.ReLU()

        self.layer2 = AutoReg(out_features, out_features, k, mask_type='B')
        self.ReLU_2 = nn.ReLU()

        self.layer3 = AutoReg(out_features + nclass, out_features, k, mask_type='B')
        self.ReLU_3 = nn.ReLU()
        self.layer4 = AutoReg(out_features, out_dim, k, mask_type='B')

        self.activation_out = activation_out

    def forward(self, x, s):

        x = self.layer1(x)
        x = self.ReLU_1(x)

        x = self.layer2(x)
        x = self.ReLU_2(x)

        # if s is not None:
        #     s = s[..., None]
        #     x = torch.cat([x, s.repeat(1, 1, x.shape[-1])], dim=1)

        x = self.layer3(x)
        x = self.ReLU_3(x)

        if self.activation_out is None:
            return self.layer4(x)

        elif self.activation_out == 'sigmoid':
            return nn.Sigmoid()(self.layer4(x))


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
