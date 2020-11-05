import argparse
import os
import datetime
import json
import copy


import yaml
from torch.utils.data.dataloader import DataLoader
from sklearn.manifold import TSNE

from source.dataset import *
from source.model import Model
from source.utils import get_logger
from source.plot import plot_swiss_roll, plot_tsne

logger = get_logger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--beta', required=True, type=float)
    parser.add_argument('--gamma', type=float)

    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    writer = copy.deepcopy(config_dict)
    writer['training'] = {}
    writer['validation'] = {}
    writer['training']['rec_loss'] = {}
    writer['validation']['rec_loss'] = {}
    writer['validation']['aud_loss'] = {}

    beta = args.beta
    config_dict['beta'] = beta

    gamma = 0
    if args.gamma is not None:
       gamma = args.gamma
    config_dict['gamma'] = gamma

    autoencoder_name = config_dict['net']['name']
    loss_name = config_dict['loss']['name']

    model = Model.from_dict(config_dict)
    batch_size = config_dict['batch_size']
    experiment = config_dict['experiment']
    n_epochs = config_dict['n_epochs']

    dataname = config_dict['data']['name']
    dset_train = globals()[dataname].from_dict(config_dict['data'], type='train')
    dset_validation = globals()[dataname].from_dict(config_dict['data'], type='test')

    train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(dset_validation, batch_size=batch_size, drop_last=True)

    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    rind = np.random.randint(0, 10 ** 5)

    chkpt_dir = f'checkpoints/{experiment}/{tstamp}_{autoencoder_name}_' \
                f'{loss_name}_{rind}'
    os.makedirs(chkpt_dir, exist_ok=True)

    model.train(train_loader, validation_loader, n_epochs,
                writer=writer, chkpt_dir=chkpt_dir)

    json_results = json.dumps(writer)

    outfolder = f'summaries/autoencoder'
    os.makedirs(outfolder, exist_ok=True)
    outfile = f'{outfolder}/{experiment}_{tstamp}_{rind}.json'
    f = open(outfile, "w")
    f.write(json_results)
    f.close()
    logger.info(f'Save results and parameters in {outfile}')
