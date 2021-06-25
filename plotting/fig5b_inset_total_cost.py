from matplotlib import pyplot as plt
import numpy as np

def plot_inset_total_cost_bars(
        figure_path,
        balancing_results
    ):

    figsize=(1,.9)
    xlim=(0,3)
    xticks= []
    xticklabels=['', '']
    yticks= [0,2,4,6]
    fontsize=7

    purple_rgb = np.array([102, 45, 145]) / 256.
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
    plt.savefig(figure_path, dpi=300)