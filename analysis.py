import argparse
import os
import yaml

from torch.utils.data.dataloader import DataLoader
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from source.plot import plot_auditor_task_all, \
    plot_rate_all, plot_bitrate_all, plot_tsne_all
from source.utils import get_logger, generate_tsne
from source.dataset import *
from source.model import Model
from source.autoencoders import *

BASELINE = {'dsprites': (0.25, 0.52, 1, 1.0, 1.0), 'compas':(0.408, 0.66, 0.12, 0.55, 0.80),
            'adult': (0.57, 0.76, 0.71, 0.71, 0.85),
            'heritage':(0.0891, 0.68, 0.12, 0.20, 0.85)}

def compute_homogeneity(Z, O, n_neighbors=10):
    """
    Get top 10 nearest neighbors for O=1 and O=0 and
    compute average distance
    :param Z:
    :param O:
    :return:
    """
    z1 = Z[O == 1]
    z0 = Z[O == 0]

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(z1)

    distance, _ = neigh.kneighbors(z0, n_neighbors, return_distance=True)
    distance_all = np.sqrt(np.sum((Z[:, None, ...] - Z[None, ...]) ** 2, -1))
    distance_all = distance_all.mean(1).mean(0)

    homogeneity = (distance / distance_all).mean(-1).mean(0)

    return homogeneity



logger = get_logger(__name__)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', required=True)

    args = parser.parse_args()

    outfolder = f'results/tests'
    os.makedirs(outfolder, exist_ok=True)

    if args.figure == 'pareto-all':
        results_list = { 'dsprites': ('a) DSprites', [
                        ('Adv', 'summaries/sweep/dsprites_adv_bitrate5'),
                        ('MMD', 'summaries/sweep/dsprites_mmd_bitrate2'),
                        (r'$\beta$-VAE', 'summaries/sweep/dsprites_vae_bitrate2'),
                        ('VFAE', 'summaries/sweep/dsprites_vfae_bitrate50'),
                        ('FBC', 'summaries/sweep/dsprites_comp_test2')
                        ]),
                        'adult':('b) Adult',
                                 [
                            ('Adv', 'summaries/sweep/adult_adv_auditor'),
                        (r'$\beta$-VAE', 'summaries/sweep/adult_vae_auditor'),
                        ('MMD', 'summaries/sweep/adult_mmd_auditor'),
                        ('VFAE', 'summaries/sweep/adult_vfae_auditor'),
                        ('FBC', 'summaries/sweep/adult_comp_auditor')
                        ]),
                        'compas': ('c) Compas', [
                        ('Adv', 'summaries/sweep/compas_adv_auditor'),
                        (r'$\beta$-VAE', 'summaries/sweep/compas_vae_auditor'),
                        ('MMD', 'summaries/sweep/compas_mmd_auditor'),
                        ('VFAE', 'summaries/sweep/compas_vfae_auditor'),
                        ('FBC', 'summaries/sweep/compas_comp_auditor')
                       ]),
                        'heritage': ('d) Heritage', [
                            ('Adv', 'summaries/sweep/heritage_adv_auditor'),
                        (r'$\beta$-VAE', 'summaries/sweep/heritage_vae_auditor'),
                       ('MMD', 'summaries/sweep/heritage_mmd_auditor'),
                        ('VFAE', 'summaries/sweep/heritage_vfae_auditor'),
                        ('FBC', 'summaries/sweep/heritage_comp_auditor')
                                     ])
        }
        plot_auditor_task_all(results_list, outfolder, tag='AAAAI',
                              baselines=BASELINE)

    if args.figure == 'rate-all':
        results_list = {'dsprites': ('a) Dsprites', 'summaries/sweep/dsprites_comp3', 300, 'probe'),
                        'adults': ('b) Adults', 'summaries/sweep/adult_comp_auditor', 20, 'probe'),
                        'compas': ('c) Compas', 'summaries/sweep/compas_comp_auditor', 20, 'probe'),
                        'heritage': ('d) Heritage', 'summaries/sweep/heritage_comp_auditor', 30, 'probe')
                        }

        plot_rate_all(results_list, outfolder, tag='AAAI', bins=20)

    if args.figure == 'bitrate-all':
        results_list = {'dsprites': ('a) Dsprites', 'summaries/sweep/dsprites_comp3', 300, 'probe'),
                        'adults': ('b) Adults', 'summaries/sweep/adult_comp_auditor', 20, 'probe'),
                        'compas': ('c) Compas', 'summaries/sweep/compas_comp_auditor', 20, 'probe'),
                        'heritage': ('d) Heritage', 'summaries/sweep/heritage_comp_auditor', 30, 'probe')
                        }

        plot_bitrate_all(results_list, outfolder, tag='CompAAAI', bins=20)

    if args.figure == 'bitrate-all-vae':
        results_list = {'dsprites': ('a) Dsprites', 'summaries/sweep/dsprites_vae_bitrate2', 300, 'probe'),
                    'adults': ('b) Adults', 'summaries/sweep/adult_vae_auditor', 20, 'probe'),
                    'compas': ('c) Compas', 'summaries/sweep/compas_vae_auditor', 20, 'probe'),
                    'heritage': ('d) Heritage', 'summaries/sweep/heritage_vae_auditor', 30, 'probe')
                    }

        plot_bitrate_all(results_list, outfolder, tag='VAEAAAI', bins=20)


    if args.figure == 'tsne-adults':

        checkpoints_list = ['checkpoints/adult_fbc_auditor/epoch_130', 'checkpoints/adult_fbc_auditor/epoch_130']
        beta_list = [0, 0.35]
        config_file = 'configs/adults/adults_fbc_tsne.yml'

        with open(config_file, 'r') as stream:
            config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

        net_dict = config_dict['net']
        name_net = net_dict.pop('name')
        net_dict.pop('learning_rate')
        net = globals()[name_net].from_dict(config_dict['net'])

        dataname = config_dict['data'].pop('name')
        dset = globals()[dataname].from_dict(config_dict['data'], type='train')

        Z_list, S_list, Y_list = generate_tsne(config_file, checkpoints_list, dset, logger)

        homo_beta_0 = compute_homogeneity(Z_list[0], S_list[0])
        homo_beta_1 = compute_homogeneity(Z_list[1], S_list[1])

        logger.info(f'Homogeneity in embedding space for S=0 and beta={beta_list[0]} is {homo_beta_0}')
        logger.info(f'Homogeneity in embedding space for S=0 and beta={beta_list[1]} is {homo_beta_1}')

        homo_beta_0_y = compute_homogeneity(Z_list[0], Y_list[0])
        homo_beta_1_y = compute_homogeneity(Z_list[1], Y_list[1])

        logger.info(f'Homogeneity in embedding space for Y=1 and beta={beta_list[0]} is {homo_beta_0_y}')
        logger.info(f'Homogeneity in embedding space for Y=1 and beta={beta_list[1]} is {homo_beta_1_y}')

        plot_tsne_all(Z_list, S_list, Y_list, outfolder, tag='AdultsAAAI_bis', nsensitive=2,
                      beta_list=beta_list, noutcome=2,
                      sensitive={0: 'Female', 1: 'Male'},
                      outcome={0: r'$<50K$', 1: r'$>50K$'})

    if args.figure == 'tsne-compas':

        checkpoints_list = ['checkpoints/compas_tsne_0/compas_comp_tsne/epoch_399',
                            'checkpoints/compas_tsne_04/compas_comp_tsne/epoch_270']
        beta_list = [0, 0.8]
        config_file = 'configs/compas/compas_comp_tsne.yml'

        with open(config_file, 'r') as stream:
            config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

        net_dict = config_dict['net']
        name_net = net_dict.pop('name')
        net_dict.pop('learning_rate')
        net = globals()[name_net].from_dict(config_dict['net'])

        dataname = config_dict['data'].pop('name')
        config_dict['data']['transfer'] = True
        dset = globals()[dataname].from_dict(config_dict['data'], type='test')

        Z_list, S_list, Y_list = generate_tsne(config_file, checkpoints_list, dset, logger)

        plot_tsne_all(Z_list, S_list, Y_list, outfolder, tag='CompasAAAI_2', nsensitive=2,
                      beta_list=beta_list, noutcome=2,
                      sensitive={1: 'African-American', 0: 'Other'},
                      outcome={0: r'Low risk', 1: r'High risk'})

    if args.figure == 'tsne-heritage':

        checkpoints_list = ['checkpoints/heritage_tsne_0/heritage_comp_tsne/epoch_20',
                            'checkpoints/heritage_tsne_12/heritage_comp_tsne/epoch_99']
        beta_list = [0, 0.5]
        config_file = 'configs/heritage/heritage_comp_tsne.yml'

        with open(config_file, 'r') as stream:
            config_dict = yaml.load(stream, Loader=yaml.SafeLoader)

        net_dict = config_dict['net']
        name_net = net_dict.pop('name')
        net_dict.pop('learning_rate')
        net = globals()[name_net].from_dict(config_dict['net'])

        dataname = config_dict['data'].pop('name')
        config_dict['data']['transfer'] = True
        dset = globals()[dataname].from_dict(config_dict['data'], type='test')

        Z_list, S_list, Y_list = generate_tsne(config_file, checkpoints_list, dset, logger, balance=False)

        plot_tsne_all(Z_list, S_list, Y_list, outfolder, tag='HeritageAAAI', nsensitive=2,
                      beta_list=beta_list, noutcome=2,
                      sensitive={1: r'$\geq 60$', 0: r'$<60$'},
                      outcome={0: r'Not morbid', 1: r'Morbid'})
