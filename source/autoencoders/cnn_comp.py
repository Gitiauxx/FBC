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


class CNNCOMP(TemplateModel):
    """implement cnn encoder-decoder as in http://proceedings.mlr.press/v97/creager19a/creager19a.pdf
    But only the mean of the distribution is learned instead. The standard deviation is fixed to sigma"""

    def __init__(self, im_shape=[64, 64], zdim=10, n_chan=1, sigma=1, k=8, sdim=1, method='compression', zk=8):
        super().__init__()
        self.encode_layers = nn.Sequential(Resize((-1, n_chan, im_shape[0], im_shape[1])),
                                           nn.Conv2d(n_chan, 32, 4, 2, 1),
                                           nn.ReLU(True), nn.Conv2d(32, 32, 4, 2, 1),
                                           nn.ReLU(True), nn.Conv2d(32, 64, 4, 2, 1),
                                           nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1),
                                           nn.ReLU(True),
                                           Resize((-1, 1024)),
                                           nn.Linear(1024, 128),
                                           nn.ReLU(True),
                                           nn.Linear(128, zk * k),
                                           nn.Tanh())

        self.feature_binary_dec = nn.Linear(zk * (k + 1), zdim)

        self.decode_layers = nn.Sequential(nn.Linear(zdim, 128),
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
        self.k = k
        self.zk = zk
        self.sigma = sigma

    def encode(self, x):
        """
        Encode a batch of images x into representations z
        :param x: B, Cin, Hin, Hout
        :return: (B, zdim), (B, zdim)
        """
        z = self.encode_layers(x)
        return z

    def binarize(self, z):
        """
        Binarize trick: quantization occurs only in the forward pass

        :param z: ;latent (B, zdim)
        :return: binary (B, zdim, k)
        """

        if len(z.shape) == 2:
            z = z.reshape(z.shape[0], self.zk, self.k)
        elif len(z.shape) == 1:
            z = z.reshape(self.zk, self.k)

        z = (z + 1) / 2
        z1 = (z - 1) ** 2
        z0 = z ** 2

        xsoft = torch.exp(-z1 * self.sigma)
        xsoft = xsoft / (torch.exp(-z1 * self.sigma) + torch.exp(-z0 * self.sigma))

        xhard = torch.round(z)

        residual = (xhard - xsoft).detach()

        return residual + xsoft

    def get_representation(self, b):
        """
        From binarized representation generate actual representatio
        :param b: (B, zdim, k)
        :return: (B, zdim)
        """
        return b.reshape(b.shape[0], self.zk * self.k)

    def decode(self, b):
        """

        :param x: (B, Cin, W, H)
        :return: (B, 3, W, H)
        """
        output = self.decode_layers(b)
        return output.squeeze()

    def forward(self, x, s):
        """
        Pass forward from input to reconstructed output
        :param x: (B, input_dim)
        :return: (B, input_dim)
        """
        z = self.encode(x)
        b = self.binarize(z)

        if len(s.shape) == 1:
            s = s[..., None]

        s = s[..., None]
        idx = torch.randint_like(s[:, 0], s.shape[0]).long()

        srand  = s[idx[:, 0], ...]
        # l = int(s.shape[0] / 2)
        # srand[:l, ...] = s[idx[:l, 0], ...]

        bk = torch.cat([s, b], -1)
        bkrand = torch.cat([srand, b.clone().detach()], -1)

        bk = bk.reshape(-1, self.zk * (self.k + 1))
        bkrand = bkrand.reshape(-1, self.zk * (self.k + 1))

        bkk = self.feature_binary_dec(bk)
        bkkrand = self.feature_binary_dec(bkrand)
        out = self.decode(bkk)
        out = (out + 1) / 2

        return out, b, [bkkrand, srand]

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
