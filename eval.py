import argparse
import datetime
import json
import os

import yaml
import numpy as np

from experiments.probe import Probe
from source.utils import get_logger
#from source.plot import plot_tsne, plot_modified_output, plot_swiss_roll

logger = get_logger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gamma', type=float)

    args = parser.parse_args()
    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    with open(args.config_path, 'r') as stream:
        config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

    seed = 0
    if args.seed is not None:
        seed = args.seed

    beta = None
    if args.beta is not None:
        beta = args.beta

    fairness = None
    if args.gamma is not None:
        fairness = args.gamma

    experiment_type = config_dict['experiment_type']
    name = config_dict['name']
    logging_dir = config_dict['logging_dir']

    checkpoint_dir = None
    if 'checkpoints' in config_dict:
        checkpoint_dir = config_dict['checkpoints']
        checkpoint_dir = checkpoint_dir + '/' + name
        os.makedirs(checkpoint_dir, exist_ok=True)

    rind = np.random.randint(0, 10**5)

    outfolder = f'{logging_dir}/pareto_front/{name}'
    os.makedirs(outfolder, exist_ok=True)
    outfile = f'{outfolder}/{tstamp}_{experiment_type}_{seed}_{rind}.json'

    if 'task_list' in config_dict:
        PR = Probe(config_dict['autoencoder'], config_dict['probes_list'],
               beta=beta,
                   seed=seed, fairness=fairness, checkpoints=checkpoint_dir,
                   task_list=config_dict['task_list'])
    else:
        PR = Probe(config_dict['autoencoder'], config_dict['probes_list'],
                   beta=beta,
                   seed=seed, fairness=fairness, checkpoints=checkpoint_dir)

    if experiment_type == 'predict_sensitive':
        PR.probe_sensitive()
    elif experiment_type == 'predict_outcome':
        PR.classify_from_representation()
    elif experiment_type == 'predict_outcome_sensitive':
        PR.probe_sensitive()
        PR.classify_from_representation()

    writer = PR.results
    json_results = json.dumps(writer)

    f = open(outfile, "w")
    f.write(json_results)
    f.close()
    logger.info(f'Save results and parameters in {outfile}')
