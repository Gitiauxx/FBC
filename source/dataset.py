import os
import math

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_swiss_roll

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as tf
from PIL import Image
from source.utils import get_logger


class AdultDataSet(Dataset):
    """
    Implement a dataset class for the adult dataset
    Sensitive Attribute: Gender if tsne, else Gender_Race

    """

    def __init__(self, filepath, range_data=None, tsne=False):

        data = pd.read_csv(filepath)

        data['workclass'] = data['workclass'].astype('category').cat.codes
        data['education'] = data['education'].astype('category').cat.codes
        data['occupation'] = data['occupation'].astype('category').cat.codes
        data['relationship'] = data['relationship'].astype('category').cat.codes
        data['marital-status'] = data['marital-status'].astype('category').cat.codes
        data['income'] = data['income'].astype('category').cat.codes
        data['gender'] = data['sex'].astype('category').cat.codes
        data['srace'] = data['race'].astype('category').cat.codes

        self.data = data[['education-num',
                           'hours-per-week',
                          'relationship', 'occupation',
                           'capital-gain', 'capital-loss',
            'marital-status', 'workclass', 'education', 'race', 'income', 'sex']]

        for var in list(self.data.columns):
            self.data = self.data[~self.data[var].isnull()]

        if range_data is not None:
            idx = np.random.choice(self.data.index, range_data, replace=True)
            self.data = self.data.loc[idx, :]
            self.data.set_index(np.arange(range_data), inplace=True)

        self.data['race_gender'] = self.data['race'] + '_' + self.data['sex']
        self.sensitive = pd.get_dummies(self.data['race_gender'])

        self.outcome = self.data['income']

        if tsne:
            self.sensitive = pd.get_dummies(self.data['sex'])

        self.data.drop(['sex', 'income', 'race', 'race_gender'], axis=1, inplace=True)

        for var in list(self.data.columns):
            mean = self.data[var].mean()
            self.data[var] = self.data[var] - mean
            std = ((self.data[var] ** 2).mean()) ** 0.5
            self.data[var] = (self.data[var]) / std

        self.tsne = tsne

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        :param: config_dict : configuration dictionary
        """

        filepath = config_data[type]
        range_data = None
        if 'range_data' in config_data:
            range_data = config_data['range_data']

        tsne = False
        if 'tsne' in config_data:
            tsne = config_data['tsne']

        return cls(filepath, range_data=range_data, tsne=tsne)

    def __len__(self):
        """
        :return: the length of data
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Overload __getitem__ with idx being an indice in self.data and self.sensitive
        :param idx:
        :return: a dictionary with torch tensor (1, 3) as input/target and torch tensor (1) as sensitive

        """
        x = torch.from_numpy(np.array(self.data.loc[idx, :])).float()
        s = torch.tensor(np.array(self.sensitive.loc[idx])).float()
        y = torch.tensor(np.array(self.outcome.loc[idx])).float()

        return {'input': x, 'target': x, 'sensitive': s, 'outcome': y}


class DSprites(Dataset):
    """
    Implement a dataset method for the DSrpites synthetic dataset
    (see https://github.com/deepmind/dsprites-dataset) but by imposing a
    correlation between shape and Xpos
    (see http://proceedings.mlr.press/v97/creager19a/creager19a.pdf)

    And outcome is whether scale > 0.75 and ((x < 0.5 && y < 0.5) & (x > 0.5 && y > 0.5))
    """

    def __init__(self, filepath, n=10000, seed=0, corr_coeff=0):

        np.random.seed(seed)

        self.corr_coeff = corr_coeff
        self.latents_sizes = np.array([1, 3, 6, 40, 32, 32])

        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                             np.array([1, ])))

        latents_sampled = self.sample_latent_corr(size=n)
        idx = np.random.choice(latents_sampled.shape[0], latents_sampled.shape[0], replace=False)

        latents_sampled = latents_sampled[idx]

        sensitives = np.zeros_like(latents_sampled[:, 3])
        sensitives[latents_sampled[:, 3] >= 10] = 1
        sensitives[latents_sampled[:, 3] >= 20] = 2
        sensitives[latents_sampled[:, 3] >= 30] = 3

        self.indices_sampled = self.latent_to_index(latents_sampled)
        self.outcomes = latents_sampled[:, 1]

        self.sensitives = np.zeros((latents_sampled[:, 3].shape[0], 4))
        self.sensitives[np.arange(latents_sampled.shape[0]), sensitives.astype('int32')] = 1

        dataset_zip = np.load(filepath, mmap_mode=None)
        self.imgs = dataset_zip['imgs']

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        :param: config_dict : configuration dictionary
        """
        if type == 'train':
            n = config_data['n_train']
        elif type == 'test':
            n = config_data['n_test']
        elif type == 'validate':
            n = config_data['n_validate']

        filepath = config_data['path']
        seed = config_data['seed']
        corr_coeff = config_data['corr_coeff']

        return cls(filepath, n=n, seed=seed, corr_coeff=corr_coeff)

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def sample_latent_corr(self, size=1):
        samples = self.sample_latent(size=size)

        p = 1 + self.corr_coeff * ((samples[:, 3][:, None] / 40) ** 3 + (np.arange(3)[None, ...] / 3) ** 3)
        p = p / p.sum(1, keepdims=True)
        s = p.cumsum(axis=1)

        r = np.random.rand(p.shape[0])
        r = r[:, None]
        posx = (s < r).sum(1)

        samples[:, 1] = posx
        return samples

    def __len__(self):
        """

        :return: the length of data
        """
        return self.indices_sampled.shape[0]

    def __getitem__(self, idx):
        """
        Overload __getitem__ with idx being an indice in self.data and self.sensitive
        :param idx:
        :return: a dictionary with torch tensor (1, 3) as input/target and torch tensor (1) as sensitive

        """
        img_idx = self.indices_sampled[idx]
        image = torch.from_numpy(np.array(self.imgs[img_idx, :])).float()
        s = torch.tensor(np.array(self.sensitives[idx])).float()
        y = torch.tensor(np.array(self.outcomes[idx])).long()

        return {'input': image, 'target': image, 'sensitive': s, 'outcome': y}


class HeritageDataset(Dataset):
    """
    Implement a dataset class for the heritage dataset
    Sensitive Attribute: Gender if Gender_Age
    Outcome: CI morbidity or Payment Delay

    """

    def __init__(self, filepath, range_data=None, tsne=False):

        data = pd.read_csv(filepath)
        if range_data is not None:
            idx = np.random.choice(data.index, size=range_data, replace=True)
            data = data.loc[idx, :]
            data.set_index(np.arange(len(data)), inplace=True)

        data['CI'] = (data['CI'] != '0').astype('int32')
        data['PayDelay'] = data['PayDelay'].astype('category').cat.codes

        self.data = data[['ClaimsCount', 'PayDelay', 'LabCount', 'DrugCount', 'CI', 'AMI',
                          'APPCHOL', 'ARTHSPIN', 'CANCRA', 'CANCRB', 'CANCRM', 'CATAST', 'CHF',
                          'COPD', 'FLaELEC', 'FXDISLC', 'GIBLEED', 'GIOBSENT', 'GYNEC1', 'GYNECA',
                          'HEART2', 'HEART4', 'HEMTOL', 'HIPFX', 'INFEC4', 'LIVERDZ', 'METAB1',
                          'METAB3', 'MISCHRT', 'MISCL1', 'MISCL5', 'MSC2a3', 'NEUMENT', 'ODaBNCA',
                          'PERINTL', 'PERVALV', 'PNCRDZ', 'PNEUM', 'PRGNCY', 'RENAL1', 'RENAL2',
                          'RENAL3', 'RESPR4', 'ROAMI', 'SEIZURE', 'SKNAUT', 'STROKE',
                          'TRAUMA', 'UTI', 'ANES', 'EM', 'MED', 'PL', 'RAD', 'SAS', 'SCS', 'SDS',
                          'SEOA', 'SGS', 'SIS', 'SMCD', 'SMS', 'SNS', 'SO', 'SRS', 'SUS', 'AgeAtFirstClaim', 'Sex',
                          ]]

        for var in list(self.data.columns):
            self.data = self.data[~self.data[var].isnull()]

        self.data.set_index(np.arange(len(self.data)), inplace=True)

        self.data['delay'] = (self.data.PayDelay > 0.5).astype('int32')

        self.data['S_A'] = self.data['Sex'] + '_' + self.data['AgeAtFirstClaim']
        self.sensitive = pd.get_dummies(self.data['S_A'])
        self.outcome = self.data['CI']

        if tsne:
            self.data['60+'] = (self.data['AgeAtFirstClaim'].isin(['80+', '70-79', '60-69'])).astype('int32')
            self.sensitive = pd.get_dummies(self.data['60+'])
            self.data.drop('60+', inplace=True, axis=1)

        self.tsne = tsne

        self.data.drop(['AgeAtFirstClaim', 'Sex', 'S_A', 'CI', 'delay'], axis=1, inplace=True)

        for var in list(self.data.columns):
            mean = self.data[var].mean()
            std = self.data[var].var() ** 0.5
            self.data[var] = (self.data[var] - mean) / std

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        :param: config_dict : configuration dictionary
        """
        if type == 'train':
            filepath = config_data['train']
        elif type == 'test':
            filepath = config_data['test']
        elif type == 'validate':
            filepath = config_data['validate']

        range_data = None
        if ('range_data' in config_data) & (type == 'train'):
            range_data = config_data['range_data']

        tsne = False
        if 'transfer' in config_data:
            tsne = config_data['tsne']

        return cls(filepath, range_data=range_data, tsne=tsne)

    def __len__(self):
        """

        :return: the length of data
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Overload __getitem__ with idx being an indice in self.data and self.sensitive
        :param idx:
        :return: a dictionary with torch tensor (1, 3) as input/target and torch tensor (1) as sensitive

        """
        x = torch.from_numpy(np.array(self.data.loc[idx, :])).float()
        s = torch.tensor(np.array(self.sensitive.loc[idx, :])).float()
        y = torch.tensor(np.array(self.outcome.loc[idx])).float()

        return {'input': x, 'target': x, 'sensitive': s, 'outcome': y}


class Compas(Dataset):

    def __init__(self, file, tsne):
        data = pd.read_csv(file)
        data = data[(data.days_b_screening_arrest <= 30) & (data.days_b_screening_arrest >= -30) &
                    (data.is_recid != -1) & (data.c_charge_degree != "O") & (data.score_text != 'N/A')]

        data['gender'] = data.sex.astype('category').cat.codes
        data['age_cat'] = data.age_cat.astype('category').cat.codes
        data['charge_degree'] = data.c_charge_degree.astype('category').cat.codes
        data['is_violent_recid'] = data.is_violent_recid.astype('category').cat.codes
        data['juv_fel_count'] = data.juv_fel_count.astype('category').cat.codes
        data['is_recid'] = data.is_recid.astype('category').cat.codes
        data['outcome'] = 2 * (data.v_score_text.isin(['High', 'Medium'])).astype('int32') - 1
        data['high_risk'] = (data.outcome == 1).astype('int32')

        self.data = data[['age_cat', 'gender', 'is_violent_recid', 'juv_fel_count', 'is_recid', 'priors_count',
                          'charge_degree', 'high_risk', 'race']]

        for var in list(self.data.columns):
            self.data = self.data[~self.data[var].isnull()]

        self.data.set_index(np.arange(len(self.data)))

        self.data['race'] = (self.data.race == 'African-American').astype('int32')
        self.data['race_gender'] = self.data['race'].astype(str) + '_' + self.data['gender'].astype(str)
        self.data['race_gender'] = self.data['race_gender'].astype('category').cat.codes
        self.sensitive = pd.get_dummies(self.data['race_gender'])
        self.outcome = self.data['high_risk']

        if tsne:
            self.sensitive = pd.get_dummies(self.data['race'])

        self.data.drop(['high_risk', 'race', 'gender', 'race_gender'], axis=1, inplace=True)

        for var in list(self.data.columns):
            mean = self.data[var].mean()
            std = self.data[var].var() ** 0.5
            self.data[var] = (self.data[var] - mean) / std

        self.tsne = tsne

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        :param: config_dict : configuration dictionary
        """

        tsne = False
        if 'tsne' in config_data:
            tsne = config_data['tsne']

        filepath = config_data[type]

        return cls(filepath, tsne=tsne)

    def __getitem__(self, idx):

        x = torch.from_numpy(np.array(self.data.loc[idx, :])).float()
        y = torch.tensor(self.outcome.loc[idx]).float()
        s = torch.tensor(self.sensitive.loc[idx]).float()

        return {'input': x, 'target': x, 'outcome': y, 'sensitive': s}

    def __len__(self):
        return len(self.data)


class SynthDataset(Dataset):

    def __init__(self, n, d=2, alpha=0, gamma=0, noise=0.1):

        x = np.zeros((n, d))
        s = np.random.randint(low=0, high=2, size=n)
        x[:, 0] = np.random.randn(n)

        for i in np.arange(1, d):
            factorization = 2 * np.random.randint(low=0, high=2, size=(n, i)) * alpha - alpha
            u = np.random.randn(n)
            v = (x[:, :i] * factorization).mean(-1)
            x[:, i] = u * (v + u > 0)

        self.x = x * (x + (2 * s[:, None] - 1) * gamma > 0)
        self.s = np.zeros((n, 2))
        self.s[:, 0] = s
        self.s[:, 1] = 1 - s
        error = np.random.randn(n)
        self.y = (x.mean(-1) + noise * error > 0).astype('int32')

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        :param: config_dict : configuration dictionary
        """
        if type == 'train':
            n = config_data['ntrain']
        elif type == 'test':
            n = config_data['ntest']
        elif type =='validate':
            n = config_data['nvalidate']

        d = config_data['d']
        gamma = config_data['gamma']
        alpha = config_data['alpha']

        return cls(n, d=d, alpha=alpha, gamma=gamma)

    def __getitem__(self, idx):

        x = torch.from_numpy(self.x[idx, ...]).float()
        y = torch.tensor(self.y[idx]).float()
        s = torch.tensor(self.s[idx, :]).float()

        return {'input': x, 'target': x, 'outcome': y, 'sensitive': s}

    def __len__(self):
        return self.x.shape[0]



class RepDataset(Dataset):
    """
    Generate represenation using the generator function that takes
    data from dset and compute generator(dset)
    """

    def __init__(self, dset, generator, adversarial=False, device='cpu',
                 batch_size=128, shuffle=False, transfer=False):
        super().__init__()
        self.dset = dset
        self.generator = generator
        self.adversarial = adversarial
        self.device = device
        self.transfer = transfer

        input_list = []
        outcome_list = []
        sensitive_list = []
        z_list = []

        data_loader = DataLoader(dset,
                                 batch_size=batch_size,
                                 shuffle=shuffle)

        for batch in data_loader:
            z = self.generate(batch['input'].to(device))
            input_list.append(batch['input'].cpu())
            z_list.append(z.cpu())

            outcome_list.append(batch['outcome'].cpu())

            if transfer:
                sensitive_list.append(batch['sensitive_transfer'].cpu())
            else:
                sensitive_list.append(batch['sensitive'].cpu())

        self.outcome = torch.cat(outcome_list, 0)
        self.sensitive = torch.cat(sensitive_list, 0)
        self.data = torch.cat(z_list, 0)
        self.input = torch.cat(input_list)

        self.zdim = z.shape[-1]

    def generate(self, x):
        """
        return represenation from x
        :return:
        """
        z = self.generator.net.encode(x)

        if self.generator.method == 'compression':
            b = self.generator.net.binarize(z).detach().cpu()
        elif self.generator.method in ['vae', 'adversarial', 'mmd']:
            b = self.generator.net.reparametrize(z[0], z[1]).detach().cpu()

        z = b.reshape(x.shape[0], -1)

        return z

    def __getitem__(self, idx):
        """
        Overload __iter__ with idx being an index value in self.indextable
        it generates as data the output of the generator encoder

        :return: an iterable
        """
        x = self.input[idx, ...]
        y = self.outcome[idx, ...]
        s = self.sensitive[idx, ...]
        z = self.data[idx, ...]

        return {'input_mean': z, 'target': y, 'sensitive': s, 'input': x}

    def __len__(self):
        """
        :return: the length of data
        """
        return len(self.dset)


