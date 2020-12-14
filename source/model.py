import torch
import numpy as np
import math

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from source.utils import get_logger, accuracy_from_logits

from source.losses import *
from source.autoencoders import *
from source.auditors import *

logger = get_logger(__name__)

torch.autograd.set_detect_anomaly(True)

class _CustomDataParallel(DataParallel):
    """
    DataParallel distribute batches across multiple GPUs

    https://github.com/pytorch/pytorch/issues/16885
    """

    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Model(object):
    """
    pytorch model with loss and neural network autoencoder/fairness auditor
    """

    def __init__(self, net, loss, pmodel=None, ploss=None, learning_rate={'autoencoder': 0.001, 'pmodel': 0.001},
                 device='cpu', beta=0, gamma=0, method='compression', auditing=0):

        device = torch.device(device)
        self.net = net.to(device)
        if method in ['compression' ,'adversarial']:
            self.pmodel = pmodel.to(device)
            self.ploss = ploss

        self.loss = loss

        self.learning_rate = learning_rate['autoencoder']
        if method in ['compression' ,'adversarial']:
            self.learning_rate_p = learning_rate['pmodel']

        self.device = device
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.auditing = auditing

        self.optimizer = torch.optim.Adam(list(self.net.parameters()), lr=self.learning_rate, betas=(0.5, 0.999),
                                              weight_decay=1e-5)

        if method in ['compression' ,'adversarial']:
            self.optimizer_pmodel = torch.optim.Adam(list(self.pmodel.parameters()), lr=self.learning_rate_p, betas=(0.5, 0.999),
                                          weight_decay=1e-5)

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a model input configuration from a config dictionary

        Parameters
        ----------
        config_dict : configuration dictionary

        """
        name_net = config_dict['net'].pop('name')
        beta = config_dict['beta']
        gamma = config_dict['gamma']
        auditing = 0
        if 'auditing' in config_dict:
            auditing = config_dict['auditing']

        method = config_dict['method']

        learning_rate = config_dict['net'].pop('learning_rate')
        lr = {'autoencoder': learning_rate}
        if method in ['compression' ,'adversarial']:
            learning_rate_p = config_dict['pmodel'].pop('learning_rate')
            lr['pmodel'] =  learning_rate_p

        config_dict['net']['method'] = method

        net = globals()[name_net].from_dict(config_dict['net'])

        if torch.cuda.device_count() > 1:
            logger.info(f'Number of gpu is {torch.cuda.device_count()}')
            net = _CustomDataParallel(net)

        name_loss = config_dict['loss'].pop('name')

        device = config_dict['device']

        if method != 'adversarial':
            config_dict['loss']['gamma'] = gamma

        if method == 'vae':
            config_dict['loss']['beta'] = config_dict['beta']

        loss = globals()[name_loss].from_dict(config_dict['loss'])

        pmodel = None
        ploss = None
        if method in ['compression', 'adversarial']:
            name_ploss = config_dict['ploss'].pop('name')
            name_pmodel = config_dict['pmodel'].pop('name')
            ploss = globals()[name_ploss].from_dict(config_dict['ploss'])
            pmodel = globals()[name_pmodel].from_dict(config_dict['pmodel'])

        model = cls(net, loss, pmodel=pmodel, ploss=ploss, learning_rate=lr,
                    device=device, beta=beta, gamma=gamma, method=method, auditing=auditing)

        return model

    def optimize_parameters(self, x, target, sensitive, auditing=True):

        """
        Optimization of both autoencoder
        :param x: input
        :param target:
        :param sensitive:
        :return:
        """

        self.optimizer.zero_grad()
        output, b, z = self.net.forward(x, sensitive)

        if self.method in ['compression' ,'adversarial']:
            self.optimizer_pmodel.zero_grad()

        if self.method in ['compression', 'adversarial']:

            if self.method == 'compression':
                loss = self.loss.forward(target, output)
                prelogits = self.pmodel.forward(b, sensitive)
                ploss = self.ploss.forward(b, prelogits.squeeze(1))
                if auditing:
                    loss = loss + self.beta * ploss

            else:
                loss = self.loss.forward(target, output)
                if len(sensitive.shape) > 1:
                    s = torch.argmax(sensitive, -1)
                else:
                    s = sensitive
                prelogits = self.pmodel.forward(b)
                ploss = self.ploss.forward(s, prelogits)

                if auditing:
                    loss = loss - self.beta * ploss

        elif self.method in ['vae', 'mmd']:
            loss = self.loss.forward(target, output, b, z, sensitive)

        loss.backward(retain_graph=True)
        self.optimizer.step()

        if self.method in ['compression' ,'adversarial']:
            ploss.backward()
            self.optimizer_pmodel.step()

        return loss

    def train(self, train_loader, validation_loader, n_epochs, writer,
              chkpt_dir=None, save=True):

        if save:
            assert chkpt_dir is not None

        for epoch in range(n_epochs):

            train_loss = 0
            auditing = (epoch >= self.auditing)

            for _, batch in enumerate(train_loader):

                input = batch['input'].to(self.device)
                target = batch['target'].to(self.device)
                sensitive = batch['sensitive'].to(self.device)

                loss = self.optimize_parameters(input, target, sensitive, auditing=auditing)
                train_loss += loss.detach().cpu() * len(input) / len(train_loader.dataset)

            writer['training']['rec_loss'][epoch] = train_loss.item()

            logger.info(f'Epoch: {epoch} Train loss: {train_loss}')

            if ((epoch % 5 == 0) | (epoch == n_epochs - 1)):
                val_loss, _, accuracy, entr_loss = self.eval(validation_loader)
                logger.info(f'Epoch: {epoch} Validation loss: {val_loss}')
                logger.info(f'Epoch: {epoch} Entropy: {entr_loss}')
                logger.info(f'Epoch: {epoch} Accuracy of context: {accuracy}')

                writer['validation']['rec_loss'][epoch] = val_loss.item()

            if (save) & ((epoch % 10 == 0) | (epoch == n_epochs -1)):
                model_dict = {'epoch': epoch,
                              'loss': self.loss,
                              'model_state_dict': self.net.state_dict(),
                              'optimizer_state_dict': self.optimizer.state_dict()}

                torch.save(model_dict, f'{chkpt_dir}/epoch_{epoch}')

                if self.method == 'compression':
                    pmodel_dict = {'epoch': epoch,
                                  'loss': self.loss,
                                  'model_state_dict': self.pmodel.state_dict(),
                                  'optimizer_state_dict': self.optimizer_pmodel.state_dict()}

                    torch.save(pmodel_dict, f'{chkpt_dir}/pmodel_epoch_{epoch}')

    def eval(self, data_loader, eps=10**(-8)):
        """
        Measure reconstruction loss for self.net and loss for self.auditor
        :param data_loader: torch.DataLoader
        :return: reconstruction loss, auditor accuracy
        """
        rec_loss = 0

        if self.method in ['compression', 'adversarial']:
            entr = 0
            mask_loss = 0
            accuracy = 0
        elif self.method == 'vae':
            entr = 0
            mask_loss = np.nan
            accuracy = np.nan
        else:
            entr = np.nan
            mask_loss = np.nan
            accuracy = np.nan

        for _, batch in enumerate(data_loader):
            input = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            sensitive = batch['sensitive'].to(self.device)

            output, b, z = self.net.forward(input, sensitive)

            output = output.detach()
            b = b.detach()

            if self.method == 'compression':
                prelogits = self.pmodel.forward(b, sensitive)

                acc = accuracy_from_logits(b, prelogits)
                accuracy += acc.detach().cpu() * len(input) / len(data_loader.dataset)

                entr_loss = self.ploss.forward(b, prelogits.squeeze(1))
                entr += entr_loss.detach().cpu() * len(input) / len(data_loader.dataset)

                # if len(sensitive.shape) > 1:
                #     sensitive = sensitive[:, None, None, ...]
                #
                #     b = b[:, :, :, None]
                #     pi_s = torch.sum(sensitive, 0)
                #     pi_s = pi_s[None, ...]
                #
                #     b0 = torch.sum(b * sensitive, 0) / (pi_s + eps)
                #     b1 = torch.sum(b * (1 - sensitive), 0) / (sensitive.shape[0] - pi_s + eps)
                #
                # else:
                #     b0 = torch.mean(b[sensitive == 0, ...], 0)
                #     b1 = torch.mean(b[sensitive == 1, ...], 0)
                #
                # m_loss = torch.mean(torch.abs(b0 - b1))
                # mask_loss += m_loss.detach().cpu() * len(input) / len(data_loader.dataset)

            elif self.method == 'adversarial':
                if len(sensitive.shape) > 1:
                    s = torch.argmax(sensitive, -1)
                    nclass = sensitive.shape[-1]
                else:
                    s = sensitive
                    nclass = 1
                prelogits = self.pmodel.forward(b)

                acc = accuracy_from_logits(s, prelogits, nclass=nclass)
                accuracy += acc.detach().cpu() * len(input) / len(data_loader.dataset)

                entr_loss = self.ploss.forward(s, prelogits)
                entr += entr_loss.detach().cpu() * len(input) / len(data_loader.dataset)

            elif self.method in ['vae', 'mmd']:
                z_mean, z_logvar = z[0], z[1]
                kl = 1 / 2 * torch.mean(torch.sum(z_mean ** 2 - z_logvar + torch.exp(z_logvar) - 1, -1))
                entr += kl.cpu().detach().numpy() * len(input) / len(data_loader.dataset)

            loss = (target - output) ** 2
            loss = torch.mean(loss.reshape(target.shape[0], -1).sum(-1))

            rec_loss += loss.cpu().detach() * len(input) / len(data_loader.dataset)

        return rec_loss, mask_loss, accuracy, entr