import numpy as np
import tensorflow as tf
from zen import ZenRNN, sgd_optimizer
from zen.components import randn_initializer, custom_initializer
from zen.components import relu
import pickle as pkl


def train_with_zen_regularization(
    task,
    N,
    l2_rate,
    initial_weights=None,
    learning_rate=1e-2,
    niter=1600
):
    # calc_regularizer = lambda \
    #     rate: no_regularizer if rate == 0 else frob_regularizer(l2_rate)

    # set RNN parameters
    clipping = {
        'method': tf.clip_by_global_norm,
        'args': (10.0,)
    }
    p = 2

    # if initial_weights is None:
    #     initializers = {
    #         'W_rec': randn_initializer(stddev=2. / np.sqrt(N)),
    #         'W_in': randn_initializer(stddev=1 / np.sqrt(N)),
    #         'W_out': randn_initializer(stddev=1 / np.sqrt(N)),
    #         'b': lambda s, _: np.zeros(s),
    #     }
    # else:
    initializers = {
        k: custom_initializer([w]) for k, w in initial_weights.items()
    }

    rnn_params = {
        'p': p,
        'num_neurons': N,
        'nonlinearity': relu,
        'zen_penalty_strength': l2_rate,
        'initializers': initializers
    }

    # make RNN
    rnn = ZenRNN(task, **rnn_params)
    train_op = sgd_optimizer(rnn, clipping=clipping)
    train_results = None

    # set training parameters
    train_args = {
        'train_op': train_op,
        'niter': niter,
        'learning_rate': learning_rate,
        'append_to': train_results,
        'trial_generator': task.generate_all_conditions,
        'save_weights': True,
    }

    # train network
    train_results = rnn.train(**train_args)

    # close tensorflow session
    rnn.close_session()

    return rnn, train_results


def train(
    data_path,
    task,
    N,
    n_iter,
    learning_rate,
    l2_reg_scale
):

    # assemble arguments for training
    initial_weights = {
        'W_rec': randn_initializer(stddev=1. / np.sqrt(N))((N, N), 0),
        'W_in': randn_initializer(stddev=1. / np.sqrt(N))((task.num_inputs, N),
                                                          0),
        'W_out': randn_initializer(stddev=1. / np.sqrt(N))(
            (N, task.num_outputs), 0),
        'b': np.zeros((N,))
    }

    train_args = [
        {
            'N': N,
            'l2_rate': l2_reg,
            'niter': n_iter,
            'initial_weights': initial_weights,
            'learning_rate': learning_rate,
        } for l2_reg in l2_reg_scale
    ]

    # call the training routine
    results = [train_with_zen_regularization(**args) for args in train_args]
    rnns, train_results = zip(*results)

    # save data
    save_data = {
        'g_norm': [
            np.linalg.norm(res['zen_imbalances'], axis=1) for res in
            train_results
        ],
        'loss': [
            res['loss'] for res in train_results
        ],
        'zen_cost': [
            res['zen_cost'] for res in train_results
        ],
        'initial_final_weights': [
            (res['weights'][0], res['weights'][-1]) for res in train_results
        ],
        'l2_reg_scale': l2_reg_scale,
    }

    with open(data_path, 'wb') as f:
        pkl.dump(save_data, f, -1)
