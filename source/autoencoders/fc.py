import torch
import math
import torch.nn as nn

from source.model_utils import OneLayerFC
from source.template_model import TemplateModel


class FC(TemplateModel):
    """
    Implement a basic fully connected vae for encoder and decoder
    """

    def __init__(self, input_dim, width=8, depth_dec=2,
                 width_dec=8, depth=2, zdim=8, sigma=0, method='compression',
                 activation="RELU", k=8, activation_out=None, sdim=1, zk=8):
        super().__init__()

        encoder_list = []
        in_dim = input_dim
        out_dim = width
        for _ in range(depth - 1):
            encoder_list.append(OneLayerFC(in_dim, out_dim, activation=activation))
            in_dim = out_dim

        self.encoder = nn.Sequential(*encoder_list)
        self.encoder_final = nn.Linear(out_dim, zdim)

        self.feature_binary = OneLayerFC(zdim, zk * k, activation='tanh')

        if method == 'compression':
            self.feature_binary_dec = nn.Sequential(OneLayerFC(zk * (k + 1), zdim, activation='RELU'))
                                                    # OneLayerFC(zdim, zdim, activation='RELU'),
                                                    # OneLayerFC(zdim, zdim, activation='RELU'),
                                                    # OneLayerFC(zdim, zdim, activation='RELU'))
        elif method == 'adversarial':
            self.feature_binary_dec = OneLayerFC(zdim, zdim, activation=activation)

        decoder_list = []
        in_dim = zdim
        out_dim = width_dec
        for _ in range(depth_dec - 1):
            decoder_list.append(OneLayerFC(in_dim, width_dec, activation=activation))
            in_dim = out_dim

        self.decoder = nn.Sequential(*decoder_list)
        if activation_out is None:
            self.decoder_final = nn.Linear(in_dim, input_dim)
        else:
            self.decoder_final = OneLayerFC(in_dim, input_dim, activation=activation_out)

        self.sigma = sigma
        self.param_init()
        self.k = k
        self.zdim = zdim
        self.zk = zk
        self.method = method

    def encode(self, x):
        """
        Encode a batch of images x into representations z
        :param x: B, input_dim
        :return: (B, zdim)
        """
        z = self.encoder(x)
        z = self.encoder_final(z)

        if self.method == 'compression':
            z = self.feature_binary(z)

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

        :param z: (B, zdim)
        :return: (B, input_dim)
        """

        output = self.decoder(b)
        output = self.decoder_final(output)

        return output

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

        return out, b.reshape(-1, self.zk * self.k), [bkkrand, srand]

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
