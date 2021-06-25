from matplotlib import pyplot as plt
import numpy as np

def plot_task_loss_with_neural_noise(
        fig_path,
        normalized_noise_levels,
        losses_orig,
        losses_balanced
    ):

    purple_rgb = np.array([102, 45, 145])/256.
    light_purple_rgb = tuple(1-.6*(1-x) for x in purple_rgb)

    m = 13
    figsize=(3,2.7)
    fontsize=8
    legendfontsize=6
    linewidth=1
    yticks= np.arange(50., step=10.)
    xticks = np.arange(.5, step=.1)
    xticklabels = ['{:.1f}'.format(v) for v in xticks]
    title='Loss on task as function of injected noise\n ' \
          'before and after balancing'
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
    ax.set_xlabel('Std. dev. of injected noise \n '
                  '(fraction of std. dev. of firing rates)',
                  fontsize=fontsize)
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
    plt.savefig(fig_path, dpi=300)
    plt.show()