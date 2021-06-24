import tensorflow as tf

class ForwardEulerCell(tf.contrib.rnn.RNNCell):
    def __init__(self, weights, n_steps=10, report_every=None, nonlinearity=tf.nn.tanh):
        """
            weights: dict, containing weight matrices
            n_steps: int, number of integration time steps
            nonlinearity: function, firing rate nonlinearity (e.g. tanh or relu)
        """
        self.W_rec = weights['W_rec']
        self.W_in = weights['W_in']
        self.b = weights['b']
        self.f = nonlinearity
        self.num_neurons = self.W_rec.get_shape().as_list()[0]
        if type(n_steps) is not int:
            raise ValueError('n_steps must be specified as int.')
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        if report_every is None:
            self.report_every = self.n_steps
        else:
            self.report_every = report_every

    @property
    def state_size(self):
        return self.num_neurons

    @property
    def output_size(self):
        return self.num_neurons

    def __call__(self, u, x, scope=None):
        """
        Args:
            u : inputs (batch_size x num_inputs)
            x : last state (batch_size x state_size)
        """
        
        # input to network and bias term
        # u, noise = inputs
        input_bias = tf.matmul(u, self.W_in) + self.b
    
        # this works:
        # for step in range(self.n_steps):
        #    x = x + self.dt*(-x + tf.matmul(self.f(x), self.W_rec) + input_bias)
        
        # replace with scan for speed?
        # def fn(x_, t):
        #     stddev = sqrt(self.dt * self.intrinsic_noise_var)
        #     noise = tf.random.normal(shape=x_.shape, mean=0., stddev=stddev)
        #     return x_ + self.dt*(-x_ + tf.matmul(self.f(x_), self.W_rec) + input_bias) + noise
        # out = tf.scan(fn, tf.range(self.report_every), initializer=x)

        def fn(x_, t):
            return x_ + self.dt*(-x_ + tf.matmul(self.f(x_), self.W_rec) + input_bias)
        out = tf.scan(fn, tf.range(self.report_every), initializer=x)

        return (self.f(out[-1]), out[-1]), out[-1]

class MorrisLecar(tf.contrib.rnn.RNNCell):
    def __init__(self, weights, gbar, vparams, n_steps=10):
        """
            weights: dict, containing weight matrices
            gbar: dict, containing maximal conductances
            vparams: dict, containing midpoints and slopes of channel activation curves
            n_steps: int, number of integration time steps
        """
        self.W_rec = weights['W_rec']
        self.W_in = weights['W_in']
        self.b = weights['b']
        
        self.gCa = gbar['ca']
        self.gK = gbar['k']
        self.gL = gbar['leak']

        self.V1 = vparams['half_ca']
        self.V2 = vparams['slope_ca']
        self.V3 = vparams['half_k']
        self.V4 = vparams['slope_k']

        self.eCa = vparams['reversal_ca']
        self.eK = vparams['reversal_k']
        self.eL = vparams['reversal_leak']

        self.M_ss = lambda v: 0.5*(1 + tf.nn.tanh((v - self.V1) / self.V2))
        self.N_ss = lambda v: 0.5*(1 + tf.nn.tanh((v - self.V3) / self.V4))
        self.tau_n = lambda v: 1 / tf.nn.cosh((v - self.V3) / (2 * self.V4))
        
        # number of neurons
        self.num_neurons = self.W_rec.get_shape().as_list()[0]
        
        if type(n_steps) is not int:
            raise ValueError('n_steps must be specified as int.')
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps

    @property
    def state_size(self):
        return 2*self.num_neurons

    @property
    def output_size(self):
        return self.num_neurons

    def __call__(self, u, x, scope=None):
        """
        Args:
            u : inputs (batch_size x num_inputs)
            x : last state (batch_size x state_size)
        Returns:
            y : output of network (batch_size x output_size)
            next_x : next state (batch_size x state_size)
        """

        # state variables
        v = x[:self.num_neurons]    # voltage
        n = x[self.num_neurons:]    # potassium activation

        # input to network and bias term
        I_in = tf.matmul(u, self.W_in) + self.b

        # todo, something faster than for loop?
        for step in range(self.n_steps):
            I_syn = tf.matmul(tf.nn.tanh(v), self.W_rec)
            I_ca = self.gCa * tf.M_ss(v) * (self.eCa - v)
            I_k = self.gK * n * (self.eK - v)
            I_leak = self.gL * (self.eL - v)
            v = v + self.dt*(I_in + I_syn + I_ca + I_k + I_leak)
            n = n + self.dt*((n - self.N_ss(v)) / self.tau_n(v)) 

        return self.f(x), x

# class DynamicCell(tf.contrib.rnn.RNNCell):
#     def __init__(self, W_rec, W_in, b,
#                        nonlinearity=tf.nn.tanh,
#                        train_recurrent_weights=True,
#                        train_input_weights=True,
#                        train_bias=True):
#         self.W_rec = tf.Variable(W_rec, name='W_rec', trainable=train_recurrent_weights)
#         self.W_in = tf.Variable(W_in, name='W_in', trainable=train_input_weights)
#         self.b = tf.Variable(b, name='b', trainable=train_bias)
#         self.f = nonlinearity

#     @property
#     def state_size(self):
#         return self._num_units

#     @property
#     def output_size(self):
#         return self._num_units

#     @property
#     def variables(self):
#         return

#     def __call__(self, u, x, scope=None):
#         """
#         Args:
#             u : (batch_size x num_inputs)
#         """

#         W_in = tf.tile(W_in, )
#         new_x = x + tf.matmul(self.f(x), self.W_rec) + tf.matmul(u, self.W_in) + self.b

#         return new_x, new_x
