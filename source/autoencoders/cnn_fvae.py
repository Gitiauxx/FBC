import torch
import math
import torch.nn as nn

from source.template_model import TemplateModel


class Resize(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CNNFFVAE(TemplateModel):
    """implement cnn encoder-decoder as in http://proceedings.mlr.press/v97/creager19a/creager19a.pdf
    But only the mean of the distribution is learned instead. The standard deviation is fixed to sigma"""

    def __init__(self, im_shape=[64, 64], zdim=10, n_chan=1, sdim=1, method='vae'):
        super().__init__()
        self.encode_layers = nn.Sequential(Resize((-1, n_chan, im_shape[0], im_shape[1])),
                                           nn.Conv2d(n_chan, 32, 4, 2, 1),
                                           nn.ReLU(True), nn.Conv2d(32, 32, 4, 2, 1),
                                           nn.ReLU(True), nn.Conv2d(32, 64, 4, 2, 1),
                                           nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1),
                                           nn.ReLU(True),
                                           Resize((-1, 1024)),
                                           nn.Linear(1024, 128),
                                           nn.ReLU(True))
        self.encoder_mean = nn.Linear(128, zdim)

        if method == 'vae':
            self.encoder_var = nn.Linear(128, zdim)

        self.decode_layers = nn.Sequential(nn.Linear(zdim + sdim, 128),
                                           nn.ReLU(True),
                                           nn.Linear(128, 1024),
                                           nn.ReLU(True), Resize((-1, 64, 4, 4)),
                                           nn.ConvTranspose2d(64, 64, 4, 2, 1),
                                           nn.ReLU(True),
                                           nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                           nn.ReLU(True),
                                           nn.ConvTranspose2d(32, 32, 4, 2, 1),
                                           nn.ReLU(True),
                                           nn.ConvTranspose2d(32, n_chan, 4, 2, 1))

        self.zdim = zdim
        self.param_init()
        self.method = method

    def encode(self, x):
        """
        Encode a batch of images x into representations z
        :param x: B, Cin, Hin, Hout
        :return: (B, zdim), (B, zdim)
        """
        z = self.encode_layers(x)
        mean = self.encoder_mean(z)

        if self.method == 'vae':
            logvar = self.encoder_var(z)
        elif self.method in ['adversarial', 'mmd']:
            logvar = torch.zeros_like(mean)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        """
       Reparametrization trick

        :param z: ;latent (B, zdim)
        :return: (B, zdim, k)
        """

        if self.method == 'vae':
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + torch.mul(std, eps)
        elif self.method in ['adversarial', 'mmd']:
            return mean

    def decode(self, b, s):
        """

        :param z: (B, zdim)
        :return: (B, input_dim)
        """
        if len(s.shape) == 1:
            s = s[..., None]

        b_s = torch.cat([b, s], -1)
        output = self.decode_layers(b_s)

        return output.squeeze()

    def forward(self, x, s):
        """
        Pass forward from input to reconstructed output
        :param x: (B, input_dim)
        :return: (B, input_dim)
        """
        z, logvar = self.encode(x)
        b = self.reparametrize(z, logvar)

        output = self.decode(b, s)

        return output, b, (z, logvar)

    def param_init(self):
        """
        Xavier's initialization
        """
        for layer in self.modules():
            if hasattr(layer, 'weight'):

                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU, nn.Tanh)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)
