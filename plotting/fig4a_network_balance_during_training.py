from matplotlib import pyplot as plt
import numpy as np

def plot_network_balance_during_training(
        fig_path,
        g_norm,
        gf_true,
        gf_shuff_rows,
        l2_reg_scale
):
    yellow_rgb = np.array([251, 176, 59]) / 256.
    purple_rgb = np.array([102, 45, 145]) / 256.
    red_rgb = np.array([193, 39, 45]) / 256.
    teal_rgb = np.array([0, 169, 157]) / 256.

    figsize = (3, 2.5)
    fontsize = 8
    legendfontsize = 6
    xticks = [0, 2000, 4000, 6000]
    yticks = [0, .5, 1, 1.5, 2, 2.5]
    linewidth = 1
    line_colors = [yellow_rgb, red_rgb, purple_rgb]
    niter = 6000
    extra_x = 500
    hist_scale = 30

    fig, ax = plt.subplots(1, figsize=figsize)

    # for res, c in zip(train_results, line_colors):
    #     g_norm = np.linalg.norm(res['zen_imbalances'], axis=1)
    for gn, c in zip(g_norm, line_colors):
        ax.plot(gn, c=c, linewidth=linewidth)
    for i, gf in enumerate(gf_shuff_rows):
        counts, bins = np.histogram(gf_shuff_rows[i], density=True)
        ax.hist(
            bins[:-1],
            bins,
            weights=hist_scale * counts,
            color=[1 - .7 * (1 - v) for v in line_colors[i]],
            #         density=True,
            orientation='horizontal',
            bottom=niter,
        )
        ax.plot(
            [niter, niter + extra_x],
            [gf_true[i], gf_true[i]],
            lw=linewidth,
            c=line_colors[i],
            linestyle=':'
        )

    ax.set_title('Network balance during training \n '
                 'with $\ell_2$ regularization ($\lambda$)', fontsize=fontsize
                 )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # ax.set_ylabel('Norm of neural gradient', fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=fontsize)

    ax.set_xlabel('Training iteration', fontsize=fontsize)
    ax.set_ylabel('Norm of neural gradients', fontsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)

    legend_labels = ['$\lambda = {}$'.format(zr) for zr in l2_reg_scale]
    legend_labels[-1] = legend_labels[-1] + ' ' * 10
    ax.legend(
        legend_labels,
        fontsize=legendfontsize,
        loc='upper left',
        bbox_to_anchor=(.55, 0, .1, .75),
        borderaxespad=0,
        frameon=False,
        ncol=1,
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
