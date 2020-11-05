import torch
import math
import torch.nn as nn

from source.model_utils import OneLayerFC
from source.template_model import TemplateModel


class FCVAE(TemplateModel):
    """
    Implement a basic fully connected vae for encoder and decoder
    """

    def __init__(self, input_dim, width=8, depth_dec=2,
                 width_dec=8, depth=2, zdim=8,
                 activation="RELU", activation_out=None, sdim=1, method='vae'):
        super().__init__()

        encoder_list = []
        in_dim = input_dim
        out_dim = width
        for _ in range(depth - 1):
            encoder_list.append(OneLayerFC(in_dim, out_dim, activation=activation))
            in_dim = out_dim

        self.encoder = nn.Sequential(*encoder_list)
        self.encoder_final = nn.Linear(out_dim, zdim)

        if method == 'vae':
            self.encoder_var = nn.Linear(out_dim, zdim)

        decoder_list = []
        in_dim = zdim + sdim
        out_dim = width_dec
        for _ in range(depth_dec - 1):
            decoder_list.append(OneLayerFC(in_dim, width_dec, activation=activation))
            in_dim = out_dim

        self.decoder = nn.Sequential(*decoder_list)
        if activation_out is None:
            self.decoder_final = nn.Linear(in_dim, input_dim)
        else:
            self.decoder_final = OneLayerFC(in_dim, input_dim, activation=activation_out)

        self.param_init()
        self.zdim = zdim
        self.method = method

    def encode(self, x):
        """
        Encode a batch of images x into representations z
        :param x: B, input_dim
        :return: (B, zdim)
        """
        z = self.encoder(x)
        zmean = self.encoder_final(z)
        if self.method == 'vae':
            log_var = self.encoder_var(z)
        elif self.method in ['adversarial', 'mmd']:
            log_var = torch.zeros_like(zmean)

        return zmean, log_var

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
        output = self.decoder(b_s)
        output = self.decoder_final(output)

        return output

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

                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.PReLU, nn.Tanh)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)