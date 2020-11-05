import json
import os
import math

import numpy as np
import torch
import h5py

import matplotlib
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
import pandas as pd

def plot_robustness(results_folder, outfolder,
                 tag='swiss_roll', rec_max=None, sigma_max=None):
    """
    Extract dp_auditor and dp_downstream and compute the
    difference.
    Plot this robustness measure as a function of beta for different values of width and
    depth.
    :param results_folder: string
    :param outfolder: string folder name where to save figures
    :param tag to add to the resulting figures
    :return:
    """
    results = []
    for filename in os.listdir(results_folder):
        with open(os.path.join(results_folder, filename)) as json_file:
            results_probe = json.load(json_file)
            sigma = results_probe['autoencoder']['beta']
            dp_auditor = results_probe['auditor_dp']
            rec_loss = results_probe['rec_loss']

            for _, probe_dict in results_probe['probes'].items():
                attacker_depth = probe_dict['depth']
                attacker_width = probe_dict['width']
                dp_probe = probe_dict['demographic_parity']

        results.append((sigma, attacker_depth, attacker_width, dp_auditor, dp_probe, rec_loss))

    results_df = pd.DataFrame(results, columns=['sigma', 'attacker_depth', 'attacker_width',
                                                'dp_auditor', 'dp_probe', 'rec_loss'])

    results_df.sort_values(by=['sigma'], inplace=True)
    results_df['robustness_gap'] = results_df['dp_probe'] - results_df['dp_auditor']

    if rec_max is not None:
        results_df = results_df[results_df.rec_loss <= rec_max]

    if sigma_max is not None:
        results_df = results_df[results_df.sigma <= sigma_max]

    results_agg = results_df.groupby(['sigma', 'attacker_depth', 'attacker_width'])[['robustness_gap']].median()
    results_agg['error'] = results_df.groupby(['sigma', 'attacker_depth', 'attacker_width'])[['robustness_gap']].var() ** 0.5
    results_agg['quantile_75'] = results_df.groupby(['sigma', 'attacker_depth', 'attacker_width'])[['robustness_gap']]\
        .quantile(0.75)
    results_agg['quantile_25'] = results_df.groupby(['sigma', 'attacker_depth', 'attacker_width'])[['robustness_gap']]\
        .quantile(0.25)
    results_agg.reset_index(inplace=True)


    fig = plt.figure(figsize=(15, 7))

    depth_choices = list((set(results_agg['attacker_depth'])))
    width_choices = list((set(results_agg['attacker_width'])))
    pltshow = []

    for depth in depth_choices:
        for width in width_choices:
            df = results_agg[(results_agg.attacker_depth == depth) & (results_agg.attacker_width == width)]
            if len(df) > 0:
                pltshow.append(plt.plot(df.sigma, df.robustness_gap, linewidth=3, label=f'Depth = {depth}, Width = {width}')[0])
                plt.fill_between(df.sigma, df.robustness_gap - df.error, df.robustness_gap + df.error,
                         label=f'Depth = {depth}, Width = {width}', alpha=0.1)

    plt.xlabel('Gaussian noise', fontsize=30)
    plt.ylabel('DP(downstream)-DP(Certificate)', fontsize=30)
    plt.legend(handles=pltshow, prop={'size': 20}, loc='upper right', title='Downstream data processors', title_fontsize=30)

    plt.axhline(y=0, lw=3, ls='--', color='black')
    plt.tick_params(labelsize=20)
    plt.savefig(f'{outfolder}/figure_icml_{tag}.png')

    plt.clf()


def read_pareto_icml(results_folder, bins=20, min_acc=0.5, dp_max=1, transfer=False):
    """
    Read file to construct pareto front between demographic parity and
    :param results_folder: folder with summaries report for each classifier
    :param bins: number of bins for demogrpahic parity
    :return: a pandas df
    """

    results = []

    for filename in os.listdir(results_folder):
        dp = 0
        accuracy = 0
        with open(os.path.join(results_folder, filename)) as json_file:
            results_probe = json.load(json_file)

            # if transfer:
            #     for epoch, probe_dict in results_probe['classifier']['0_transfer']['validation'].items():
            #         dp = probe_dict['dp']
            #         accuracy = probe_dict['accuracy']
            #
            #         results.append((dp, accuracy))
            #
            # else:
            if 'validation' in results_probe['classifier']['0']:
                for epoch, probe_dict in results_probe['classifier']['0']['validation'].items():

                        #if probe_dict['accuracy'] > accuracy:
                        dp = probe_dict['dp']
                        accuracy = probe_dict['accuracy']

                        results.append((dp, accuracy))
            else:
                for epoch, probe_dict in results_probe['classifier']['validation'].items():

                        #if probe_dict['accuracy'] > accuracy:
                        dp = probe_dict['dp']
                        accuracy = probe_dict['accuracy']

                        results.append((dp, accuracy))

            #for epoch, probe_dict in results_probe['classifier']['0'].items():
            #dp = results_probe['classifier']['0']['demographic_parity']
            #accuracy = results_probe['classifier']['0']['accuracy']

            #results.append((dp, accuracy))

    results_df = pd.DataFrame(results, columns=['dp', 'accuracy'])

    results_df = results_df[~results_df.dp.isnull()]
    results_df = results_df[~results_df.accuracy.isnull()]

    results_df['bins'] = pd.cut(results_df.dp, bins=bins, labels=False)
    results_df = results_df.sort_values(by='bins')
    results_df = results_df[~results_df.bins.isnull()]
    results_df = results_df[results_df.dp <= dp_max]
    results_agg = results_df.groupby('bins').size().to_frame("number")
    results_agg['dp'] = results_df.groupby('bins').dp.median()
    results_agg['accuracy'] = results_df.groupby('bins').accuracy.quantile(0.75)

    return results_agg

def plot_pareto_icml(results_list, outfolder, tag=None, bins=40, dp_max=0.2, min_yacc=0.5, transfer=False):
    """
    Plot a pareto front between demographic parity and
    :param results_list: list of tuple (name, directory of summary)
    :param outfolder:
    :param tag:
    :return:
    """
    colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0']

    pltshow = []
    fig = plt.figure(figsize=(15, 7))
    plt.axhline(y=min_yacc, color='black', linestyle='-.', linewidth=2.5)
    plt.text(0.05, min_yacc + 0.005, r'Random choice for $Y$', fontsize=20)

    for i, method in enumerate(results_list):
        name = method[0]
        results = read_pareto_icml(method[1], bins=bins, min_acc=min_yacc, dp_max=dp_max, transfer=transfer)
        results = results[results.dp <= dp_max]
        pltshow.append(plt.plot(results.dp, results.accuracy, linewidth=2.5, label=f'{name}', color=colors[i])[0])
        #pltshow.append(plt.scatter(results.dp, results.accuracy, linewidth=2.5, label=f'{name}'))
        del results

    plt.xlabel('Demographic Disparity', fontsize=20)
    plt.xticks(fontsize=20)

    plt.ylabel('Accuracy', fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(handles=pltshow, prop={'size': 20}, loc='lower right',
               title_fontsize=20)

    if transfer:
        tag = f'{tag}_transfer'

    plt.savefig(f'{outfolder}/pareto_front_{tag}.png')

    plt.clf()


def plot_swiss_roll(s, z, outfolder, tag=None):
    """
    Plot Swiss Roll in a 2d plane, where x is generated with smoothing beta
    if beta is not None; else x is original data.
    Plot the corresponding representation
    :param x: (B, 3)
    :param s: (B)
    :param z: (B, 2) representations
    :param outfolder: string folder name where to save figure
    :return:
    """
    fig = plt.figure(figsize=(10, 5))
    colors = ["#85c1e9", "#f5b041"]

    axs0show = list(range(2))
    for i in range(2):
        axs0show[i] = plt.plot(z[s == i, 0], z[s == i, 1], color=colors[i], marker="*", alpha=0.3,
                              linewidth=0, label=f'Sensitive Attribute S={i}')[0]

    plt.xlabel('x1', fontsize=15)
    plt.ylabel('x2', fontsize=15)
    plt.legend(handles=axs0show, prop={'size': 15}, loc='upper right')

    if tag is None:
        plt.savefig(f'{outfolder}/swiss_roll_original.png')
    else:
        plt.savefig(f'{outfolder}/swiss_roll_beta_{tag}.png')


def plot_tsne(tsne_results, s, outfolder, cmap_name='tab10', tag=None, title=None):
    """
    t-sne 2d scatter plot colored by sensitive attributes s
    :param tnse_results: (input_ddim, 2)
    :param s: (input_dim)
    :param outfolder:
    :param tag:
    :return:
    """

    fig = plt.figure(figsize=(10, 5))
    if title is not None:
        fig.suptitle(f'{title}', fontsize=20)

    gs = gridspec.GridSpec(4, 4)

    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_y = fig.add_subplot(gs[1:4, 3])

    label = np.unique(s)

    cmap = plt.cm.get_cmap(cmap_name, label.shape[0])

    ax_joint.scatter(tsne_results[:, 0], tsne_results[:, 1], c=s, alpha=0.3, cmap=cmap)
    ax_joint.set_xlabel('t-sne first component', fontsize=15)
    ax_joint.set_ylabel('t-sne second component', fontsize=15)

    #plt.legend(prop={'size': 15}, loc='upper right', title='Sensitive Attribute')
    for i in range(label.shape[0]):
        ax_marg_x.hist(tsne_results[s == i, 0], bins=100, color=cmap(i)[:3], alpha=0.3, density=True)
        ax_marg_y.hist(tsne_results[s == i, 1], bins=100, color=cmap(i)[:3], orientation="horizontal", alpha=0.3, density=True)

    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    if tag is None:
        plt.savefig(f'{outfolder}/t_sne_plot.png')
    else:
        plt.savefig(f'{outfolder}/t_sne_plot_{tag}.png')

def plot_tsne_all(results_list, s_list, y_list, outfolder, nsensitive=4, tag=None,
                  beta_list=[0], noutcome=3, sensitive=None, outcome=None):
    """
    plot tsne 2 d components from results list per value of sensitive attribute
    :param results_list:
    :param outfolder:
    :return:
    """

    fig, ax = plt.subplots(figsize=(10, 5), nrows=2, ncols=len(results_list))

    cmap = ['#ffe945',"#00204c", '#1e8449', '#d35400']
    c2 = plt.cm.get_cmap('cividis', nsensitive)
    #orientation = {0: r'$[0, \pi/2]$', 1: r'$[\pi/2, \pi]$', 2: r'$[\pi, 3\pi/2]$', 3: r'$[3\pi/2, \pi]$'}

    for i, tsne in enumerate(results_list):
        s = s_list[i]

        for j in range(nsensitive):
            ax[0, i].scatter(tsne[s == j, 0], tsne[s == j, 1], color=cmap[j], alpha=0.5, label=sensitive[j], s=10)

        ax[0, i].tick_params(axis='both', which='major', labelsize=15)
        ax[0, i].set_title(r'$\beta={}$'.format(beta_list[i]), size=20)

    cmap = ['#884ea0', '#f4d03f', '#196f3d', ]
    #shapes = {0:'square', 1:'ellipse', 2:'heart'}
    for i, tsne in enumerate(results_list):
        y = y_list[i]

        for j in range(noutcome):
            ax[1, i].scatter(tsne[y == j][:, 0], tsne[y == j][:, 1], color=cmap[j], s=15, alpha=0.5, label=outcome[j])

        ax[1, i].tick_params(axis='both', which='major', labelsize=15)

    fig.subplots_adjust(top=0.9, left=0.1, right=0.8,
                        bottom=0.1)


    #kw = dict(prop='colors', func=lambda s: np.array([orientation[s[i]] for i in range(s.shape[0])]))

    leg1 = ax[0, -1].legend(loc='upper left',
                           bbox_to_anchor=(1.01, 1), prop={'size': 12}, title=r'$S$: Race', title_fontsize=15)
    leg1.get_frame().set_linewidth(0.0)
    leg1._legend_box.align = "left"

    leg2 = ax[1, -1].legend(loc='upper left',
                     bbox_to_anchor=(1.01, 1), prop={'size': 12}, title=r'$Y$: Outcome', title_fontsize=15)
    leg2.get_frame().set_linewidth(0.0)

    leg2._legend_box.align = "left"

    fig.text(0.5, 0.01, r't-sne first component', fontsize=20, ha='center')
    fig.text(0.02, 0.5, r't-sne second component', fontsize=20, va='center', rotation='vertical')
    plt.savefig(f'{outfolder}/t_sne_plot_{tag}.png')


def read_pareto_auditor_task(results_folder, bins=12, as_min=None):
    """
    Read file to construct pareto front between accuracy of adversary and accuracy
    of downstream user
    :param results_folder: folder with summaries report for each classifier
    :param bins: number of bins for demogrpahic parity
    :return: a pandas df
    """

    results = []

    for filename in os.listdir(results_folder):
        adv = 0
        accuracy = 0
        with open(os.path.join(results_folder, filename)) as json_file:
            results_probe = json.load(json_file)

            if 'validation' in results_probe['classifier']['0']:
                for epoch, probe_dict in results_probe['classifier']['0']['validation'].items():

                    if probe_dict['accuracy'] > accuracy:
                        accuracy = probe_dict['accuracy']
            else:
                for epoch, probe_dict in results_probe['classifier']['validation'].items():

                    if probe_dict['accuracy'] > accuracy:
                        accuracy = probe_dict['accuracy']

            for epoch, probe_dict in results_probe['probes']['validation'].items():

                if probe_dict['accuracy'] > adv:
                    adv = probe_dict['accuracy']

            results.append((adv, accuracy))

    results_df = pd.DataFrame(results, columns=['accuracy_auditor', 'accuracy_task'])

    results_df = results_df[~results_df.accuracy_auditor.isnull()]
    results_df = results_df[~results_df.accuracy_task.isnull()]

    if as_min is not None:
        results_df = results_df[results_df.accuracy_task > as_min]

    results_df['bins'] = pd.cut(results_df.accuracy_auditor, bins=bins, labels=False, include_lowest=True)
    results_df = results_df.sort_values(by='bins')
    results_df = results_df[~results_df.bins.isnull()]
    results_agg = results_df.groupby('bins').size().to_frame("number")
    results_agg = results_agg[results_agg.number > 0]
    results_agg = results_agg[~results_agg.number.isnull()]
    results_agg['accuracy_auditor'] = results_df.groupby('bins').accuracy_auditor.median()
    results_agg['accuracy_task'] = results_df.groupby('bins').accuracy_task.quantile(0.75).cummax()

    return results_df


def plot_auditor_task_all(results_list, outfolder, tag=None, baselines=None):
    """
    Plot fairness-accuracy on separate plot, each for a given dataset
    :param results_list:
    :param outfolder:
    :param tag:
    :param baselines:
    :return:
    """

    pltshow = []
    fig, ax = plt.subplots(nrows=1, ncols=len(results_list), figsize=(16, 5))
    colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0']

    pointer = 0
    for data, res_list in results_list.items():
        dataname = res_list[0]
        mlp_accuracy = baselines[data][4]
        mlp_auditor = baselines[data][3]
        ax[pointer].scatter(mlp_auditor, mlp_accuracy, marker='s', s=50, label=r'MLP', c='#626567')

        for i, method in enumerate(res_list[1]):
            name = method[0]
            if data in ['dsprites', 'heritage', 'compas', 'adult']:
                if data == 'compas':
                    results = read_pareto_auditor_task(method[1], as_min=0.67)
                else:
                    results = read_pareto_auditor_task(method[1])

                ax[pointer].scatter(results.accuracy_auditor, results.accuracy_task,
                label=f'{name}', color=colors[i])

            else:
                results = read_pareto_icml(method[1], bins=10)
                results['dp'] = results['dp'] + baselines[data][0]
                results = results[results.dp <= baselines[data][2]]
                ax[pointer].plot(results.dp, results.accuracy,
                                 linewidth=2.5, label=f'{name}', color=colors[i])

        ax[pointer].tick_params(axis='both', which='major', labelsize=15)
        ax[pointer].set_title(f'{dataname}', size=20)

        min_yacc = baselines[data][1]
        min_sacc = baselines[data][0]
        line1 = ax[pointer].axhline(y=min_yacc, color='black', linewidth=2.5, label=r'Chance level for $y$')
        line1.set_dashes([2, 2, 10, 2])
        ax[pointer].axvline(x=min_sacc, color='black', dashes=(5, 5), linewidth=2.5, label=r'Chance level for $s$')


        pointer += 1

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
                        bottom=0.2)
    ax.flatten()[2].legend(loc='upper center',
                           bbox_to_anchor=(0.0, -0.15), ncol=8, prop={'size': 15})

    fig.text(0.5, 0.1, r'Auditor accuracy $A_{s}$', fontsize=15, ha='center')
    fig.text(0.06, 0.5, r'Task accuracy $A_{y}$', fontsize=15, va='center', rotation='vertical')

    plt.savefig(f'{outfolder}/pareto_auditor_task_{tag}.png')

    plt.clf()


def formating_to_rate(results_folder, bins=10, br_max=200, type='probe'):

    results = []
    for filename in os.listdir(results_folder):

        with open(os.path.join(results_folder, filename)) as json_file:
            results_probe = json.load(json_file)

            if type == 'task':
                dp = results_probe['classifier']['0']["demographic_parity"]
            elif type == 'probe':
                dp = results_probe['probes']['0']["accuracy"]

            br = results_probe['validation']["bit_rate"]
            rec_loss = results_probe['validation']['rec_loss_final']
            results.append((dp, br, rec_loss))

    results_df = pd.DataFrame(results, columns=['dp', 'br', 'rec_loss'])

    if br_max is not None:
        results_df = results_df[results_df.br <= br_max]


    results_df = results_df[results_df.br > 10 ** (-3)]

    results_df['bins'] = pd.cut(results_df.br, bins=bins, labels=False)
    results_df['number'] = results_df.groupby('bins').br.transform("size")
    results_df = results_df[results_df.number > 3]

    results_df['dp_95'] = results_df.groupby('bins').dp.transform(lambda gr: np.quantile(gr, 0.95))
    results_df['dp_5'] = results_df.groupby('bins').dp.transform(lambda gr: np.quantile(gr, 0.05))
    results_df = results_df[(results_df.dp <= results_df.dp_95) & (results_df.dp >= results_df.dp_5)]

    results_df['rec_95'] = results_df.groupby('bins').rec_loss.transform(lambda gr: np.quantile(gr, 0.95))
    results_df['rec_5'] = results_df.groupby('bins').rec_loss.transform(lambda gr: np.quantile(gr, 0.05))
    results_df = results_df[(results_df.rec_loss <= results_df.rec_95) & (results_df.rec_loss >= results_df.rec_5)]

    return results_df

def plot_rate_all(results_list, outfolder, bins=20, tag=None):

    """
    plot rate-fairness and distortion for each experiments in results_list
    :param results_list:
    :param outfolder:
    :param bins:
    :param tag:
    :param br_max:
    :return:
    """

    fig, ax = plt.subplots(nrows=1, ncols=len(results_list), figsize=(16, 4))
    colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0']
    ax2list = []

    pointer = 0

    for data, res_list in results_list.items():
        dataname = res_list[0]
        results_folder = res_list[1]
        task_type = res_list[3]

        br_max = res_list[2]

        results_df = formating_to_rate(results_folder, bins=bins, br_max=br_max, type=task_type)

        ax[pointer].scatter(results_df.br, results_df.dp, linewidth=2.5,
                                   label=r'Auditor accuracy $A_{s}$', color=colors[-1])

        ax2 = ax[pointer].twinx()
        ax2.scatter(results_df.br, results_df.rec_loss, linewidth=2.5, color=colors[2], label=r'Distortion')

        ax[pointer].tick_params(axis='both', which='major', labelsize=15)
        ax[pointer].set_title(f'{dataname}', size=20)
        ax[pointer].set_xlabel(r'Bitrate (nats)', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)

        ax2list.append(ax2)

        pointer += 1

    fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
                        bottom=0.25)
    ax.flatten()[1].legend(loc='upper center', frameon=False,
                           bbox_to_anchor=(0.5, -0.25), ncol=1, prop={'size': 18})
    ax2list[2].legend(loc='upper center', frameon=False,
                           bbox_to_anchor=(0.5, -0.25), ncol=1, prop={'size': 18})

    fig.text(0.01, 0.5, r'Task accuracy $A_{y}$', fontsize=15, va='center', rotation='vertical')
    fig.text(0.985, 0.5, r'L2 loss', fontsize=15, va='center', rotation='vertical')

    fig.tight_layout(rect=[0.01, 0.0, 0.99, 1])

    plt.savefig(f'{outfolder}/rate_distortion_{tag}_all.png')


def format_bitrate(results_folder, br_max=None, bins=20, type='task'):
    results = []

    for filename in os.listdir(results_folder):

        with open(os.path.join(results_folder, filename)) as json_file:
            results_probe = json.load(json_file)

            if type == 'task':
                dp = results_probe['classifier']['0']["demographic_parity"]
            elif type == 'probe':
                dp = results_probe['probes']['0']["accuracy"]

            br = results_probe['validation']["bit_rate"]
            beta = results_probe['beta']

            rec_loss = results_probe['validation']['rec_loss_final']
            results.append((dp, br, rec_loss, beta))

    results_df = pd.DataFrame(results, columns=['dp', 'br', 'rec_loss', 'beta'])
    if br_max is not None:
        results_df = results_df[results_df.br <= br_max]

    results_df['rec_loss'] = results_df['rec_loss']
    results_df = results_df[results_df.br > 10 ** (-3)]

    results_df['bins'] = pd.cut(results_df.br, bins=bins, labels=False)
    results_df['number'] = results_df.groupby('bins').br.transform("size")
    results_df = results_df[results_df.number > 3]

    results_df['dp_95'] = results_df.groupby('bins').dp.transform(lambda gr: np.quantile(gr, 0.95))
    results_df['dp_5'] = results_df.groupby('bins').dp.transform(lambda gr: np.quantile(gr, 0.05))
    results_df = results_df[(results_df.dp <= results_df.dp_95) & (results_df.dp >= results_df.dp_5)]

    results_df = results_df.sort_values(by='bins')
    results_df_for_agg = results_df[~results_df.bins.isnull()]

    results_agg = results_df_for_agg.groupby('bins').size().to_frame("number")
    results_agg['br'] = results_df_for_agg.groupby('bins').br.mean()
    results_agg['rec_loss'] = results_df_for_agg.groupby('bins').rec_loss.min()

    results_agg.sort_values(by='br', inplace=True)
    results_agg['cum_rec_loss'] = results_agg['rec_loss'].cummin()

    results_agg['dp'] = results_df_for_agg.groupby('bins').dp.min()
    results_agg.sort_values(by='br', inplace=True, ascending=False)
    results_agg['cum_dp'] = results_agg['dp'].cummin()
    results_agg.sort_values(by='br', inplace=True, ascending=True)

    return results_df, results_agg

def plot_bitrate_all(results_list, outfolder, tag=None, bins=20):
    """
    plot bit rate versus beta for each dataset
    :param results_list:
    :param outfolder:
    :param tag:
    :param br_max:
    :return:
    """

    fig, ax = plt.subplots(nrows=1, ncols=len(results_list), figsize=(16, 4))
    colors = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0']
    ax2list = []

    pointer = 0

    for data, res_list in results_list.items():
        dataname = res_list[0]
        results_folder = res_list[1]
        task_type = res_list[3]

        br_max = res_list[2]

        results_df, results_agg = format_bitrate(results_folder, br_max=br_max, type=task_type, bins=bins)

        ax1 = ax[pointer].scatter(results_df.br, results_df.dp, linewidth=2.5, c=results_df.beta, cmap=plt.cm.coolwarm)
        cbar = fig.colorbar(ax1, ax=ax[pointer], orientation='horizontal', pad=0.3)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(r'$\beta$', fontsize=15)

        ax[pointer].tick_params(axis='both', which='major', labelsize=15)
        ax[pointer].set_title(f'{dataname}', size=20)
        ax[pointer].set_xlabel(r'Bitrate (nats)', fontsize=15)

        pointer += 1

    fig.text(0.06, 0.6, r'Auditor accuracy $A_{s}$', fontsize=15, va='center', rotation='vertical')

    plt.savefig(f'{outfolder}/bitrate_{tag}_all.png')












