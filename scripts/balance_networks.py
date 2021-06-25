
import rnnops
import pickle as pkl

from os.path import join
from scipy.io import loadmat
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import gaussian_filter1d


### LOAD DATA
data_dir = '../data/neural-gradients-regularized-training/'
weight_fname = 'imbalances-during-regularized-training-03.pkl'
trial_fname = 'integration_trials.pkl'

with open(join(data_dir, weight_fname), 'rb') as f:
    weight_data = pkl.load(f)
with open(join(data_dir, trial_fname), 'rb') as f:
    trial_data = pkl.load(f)    


### 
weight_list = weight_data['initial_final_weights']
rnns = []

for i, weights in enumerate(weight_list):
    wf=weights[1]

    rnn = rnnops.RNN(
        w_in=wf['W_in'].T,
        w_out=wf['W_out'].T,
        w_rec=wf['W_rec'].T,
        b=wf['b'],
        nonlinearity='relu',
        name='trained M-S network, l2 reg {}'.format(weight_data['l2_reg_scale'][i])
    )
    rnns.append(rnn)
    print(rnn)

num_trials = len(trial_data)
T = trial_data[0][0].shape[1]
dt = .1
tt = np.arange(T, step=dt)


def repeat_and_smooth(trial_data):
    return gaussian_filter1d(np.repeat(trial_data, int(1/dt), axis=0), sigma=5, axis=0)

def concat_trials(x):    
    return np.moveaxis(np.array(x), 0, -1)

inputs, targets = [], []
for input_data, target_data, _, _ in trial_data:
    inp = repeat_and_smooth(np.moveaxis(input_data, 0, -1))
    tar = repeat_and_smooth(target_data.T)
    inputs.append(inp)
    targets.append(tar)
inputs = concat_trials(inputs)
targets = concat_trials(targets)

num_train_trials = 300
num_eval_trials = 300
    
def create_trials(batch_slice):
    trial = rnnops.trial.Trial(
        trial_len = T,
        dt=dt,
        name='Mante Sussillo all conditions',
        inputs=inputs[:,:,:,batch_slice],
        targets=targets[:,:,:,batch_slice]
    )
    return trial


train_trials = create_trials(slice(0,num_train_trials))
eval_trials = create_trials(slice(num_train_trials,num_train_trials+num_eval_trials))

print(train_trials, train_trials.shape(1))
print(eval_trials, eval_trials.shape(1))


# run original network with and without nosie
i=0
rnn_orig = rnns[i]
noise_levels = np.arange(0.7, step= 0.05)

train_trials_orig = rnnops.trial.run_neural_dynamics(
    rnn_orig,
    train_trials,
    noise_std=0)

eval_trials_orig = [
    rnnops.trial.run_neural_dynamics(
        rnn_orig,
        eval_trials,
        noise_std=s
    ) for s in noise_levels]


### RUN BALANCING

from rnnops.ops.balancing import robustness_cost_fn
from rnnops.ops.balancing import solve_balancing

T_max=1000

cost_fn = robustness_cost_fn(
    train_trials_orig,
    nonlinearity='relu',
    weight_by_velocity=False,
)

rnn_balanced, opt_results = solve_balancing(
    rnn_orig,
    cost_fn=cost_fn,
    how='odeint',
    method='RK45',
    T_max=T_max,
)
print({k: opt_results[k] for k in ['c0', 'cf']})

# run balanced network on evaluation trials

eval_trials_balanced = [
    rnnops.trial.run_neural_dynamics(
        rnn_balanced,
        eval_trials,
        noise_std=s
    ) for s in noise_levels]


### COMPARE TRIAL-AVERAGED RESPONSES


### SAVE DATA FOR PLOTTING

# save trial data

trial_data = {
    'trials_orig_mean': trials_orig_mean,
    'trials_balanced_mean': trials_balanced_mean,
    'noise_levels': noise_levels,
    'balancing_results': opt_results,
    'x_std_dev': x_std_dev,
}

trial_data_path = join(data_dir, 'noisy_trials_03_4.pkl')
with open(trial_data_path,'wb') as f:
    pkl.dump(trial_data, f, -1)