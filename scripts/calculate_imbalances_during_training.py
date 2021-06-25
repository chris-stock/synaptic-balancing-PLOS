import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

####  DATA PATHS
THIS = 'N-500_ctx-noise-0.1_input-noise-0.1'
data_dir = '../data/neural-gradients-regularized-training/'
fig_dir = '../figures_new/trained-networks'
fig_path='g-norm-regularized-training-{}.pdf'.format(THIS)
fname = os.path.join(data_dir, 'imbalances-during-regularized-training-{}.pkl'.format(THIS))


### LOAD DATA
with open(fname, 'rb') as f:
    data = pkl.load(f)
    
train_results = [{'weights': w} for w in data['initial_final_weights']]
g_norm = data['g_norm']
l2_reg_scale = data['l2_reg_scale']
print("Loaded data: {}".format(fname))

## HELPER FUNCTIONS

yellow_rgb = np.array([251, 176, 59])/256.
purple_rgb = np.array([102, 45, 145])/256.
red_rgb = np.array([193, 39, 45])/256.
teal_rgb = np.array([0, 169, 157])/256.


def calc_neural_gradient(C):
    return np.sum(C - C.T, axis=1)

def calc_g_norm(J, shuffle=False, rows=True):
    C = J**2
    if shuffle:
        C = np.random.permutation(C) if rows else np.random.permutation(C.T).T
    g = calc_neural_gradient(C)
    return np.linalg.norm(g)


### SHUFFLE ROWS    
n_shuff = int(1e4)
gf_shuff_rows = [
    [calc_g_norm(res['weights'][-1]['W_rec'], shuffle=True, rows=True)
     for _ in range(n_shuff)]
    for res in train_results]
g0_shuff_rows = [
    [calc_g_norm(res['weights'][0]['W_rec'], shuffle=True, rows=True)
     for _ in range(n_shuff)]
    for res in train_results]

gf_true = [
    calc_g_norm(res['weights'][-1]['W_rec'], shuffle=False)
    for res in train_results
]

### PLOT IMBALANCES OVER LEARNING VS. SHUFFLED IMBALANCES
figsize=(3,2.5)
fontsize=8
legendfontsize=6
xticks = [0, 2000, 4000, 6000]
yticks = [0, .5, 1, 1.5, 2, 2.5]
linewidth=1
line_colors = [yellow_rgb, red_rgb, purple_rgb]
niter=6000
extra_x=500
hist_scale=30 

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
        weights=hist_scale*counts,
        color=[1 - .7*(1-v) for v in line_colors[i]],
#         density=True,
        orientation='horizontal',
        bottom=niter,
    )    
    ax.plot(
        [niter,niter+extra_x],
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

legend_labels=['$\lambda = {}$'.format(zr) for zr in l2_reg_scale]
legend_labels[-1] = legend_labels[-1]+' '*10
ax.legend(
    legend_labels,
    fontsize=legendfontsize,
    loc='upper left',
    bbox_to_anchor=(.55,0,.1,.75),
    borderaxespad=0,
    frameon=False,  
    ncol=1,
)

plt.tight_layout()
plt.savefig(
    os.path.join(fig_dir, fig_path),
    dpi=300
)

print("Saved figure: {}".format(os.path.join(fig_dir, fig_path)))
