import os
from os.path import join
import rnnops
import pickle as pkl

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm


### LOAD DATA
THIS = 'N-500_ctx-noise-0.1_input-noise-0.1'
data_dir = '../data/neural-gradients-regularized-training/'
fig_dir ='../figures_new/trained-networks/'
noise_performance_fname = 'loss_as_fn_of_noise_orig_balanced_04.pdf'
cost_bars_fname = 'total_cost_comparison_03.pdf'

data_fname = ''
with open(os.path.join(data_dir, fname), 'rb') as f:
    data = pkl.load(f)


### CALCULATE TRIAL-AVERAGED MEAN
from rnnops.trial import trial_apply

trials_orig_mean = data['trials_orig_mean']
trials_balanced_mean = data['trials_balanced_mean']
noise_levels = data['noise_levels']
balancing_results = data['balancing_results']
x_std_dev = data['x_std_dev']


losses_orig = [calc_loss(tr) for tr in trials_orig_mean]
losses_balanced = [calc_loss(tr) for tr in trials_balanced_mean]

normalized_noise_levels = noise_levels/x_std_dev


### PLOTTING CODE
def calc_loss(trial):
    return np.mean((trial.targets - trial.outputs)**2)

def plot_performance(fname):
    purple_rgb = np.array([102, 45, 145])/256.
    light_purple_rgb = tuple(1-.6*(1-x) for x in purple_rgb)

    m = 13
    figsize=(3,2.7)
    fontsize=8
    legendfontsize=6
    linewidth=1
    yticks= np.arange(50., step=10.)
    # xticks=np.arange(2.5, step=.5)
    xticks = np.arange(.5, step=.1)
    xticklabels = ['{:.1f}'.format(v) for v in xticks]
    # xticks = np.arange(1.5, step = .5)
    title='Loss on task as function of injected noise\n before and after balancing'
    # xlim = (0,3)
    # ylim = (0, 100)
    # xlim = (0,2)
    # ylim = (0, 60)


    fig, ax = plt.subplots(
        1, 
        figsize=figsize,
    )

    ax.plot(
        normalized_noise_levels[:m],
    #     noise_levels[:m],
        losses_orig[:m],
        c=purple_rgb,
        linestyle='-',
        lw=linewidth,
    )
    ax.plot(
        normalized_noise_levels[:m],
    #     noise_levels[:m],    
        losses_balanced[:m],
        c=light_purple_rgb,
        linestyle='-',
        lw=linewidth,
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # ax.set_xlim(xlim)
    ax.set_xlabel('Std. dev. of injected noise \n (fraction of std. dev. of firing rates)', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(5.))
    ax.set_xticklabels(xticklabels, fontsize=fontsize)
    ax.set_yticklabels(yticks.astype(int), fontsize=fontsize)


    ax.legend(
        ['original network', 'balanced network'],
        fontsize=legendfontsize,
        frameon=False,   
        loc='lower right',
        ncol=1,
        borderaxespad=0,
    )

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(join(fig_dir, fname), dpi=300)
    plt.show()


def plot_cost_bars(fname):

    figsize=(1,.9)
    xlim=(0,3)
    # xticks = [1, 2]
    xticks= []
    # xticklabels=['orig.', 'balanced']
    xticklabels=['', '']
    yticks= [0,2,4,6]

    fontsize=7

    light_purple_rgb = tuple(1-.6*(1-x) for x in purple_rgb)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.bar(
        [1, 2],
        [balancing_results['c0'], balancing_results['cf']],
        color=[purple_rgb, light_purple_rgb]
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_xlim(xlim)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.))
    ax.set_ylabel('Total cost $C$', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(join(fig_dir, fname), dpi=300)
    plt.show() 


plot_performance(noise_performance_fname)
plot_cost_bars(cost_bars_fname)
