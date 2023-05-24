#%%
import time
import pickle as pic
import numpy as np
import MNIST_TOOLS_ICML as mt
from numpy.random import Generator, PCG64

import wandb as wb


def sigmoid(x):
    """
    element-wise sigmoid
    :param x:
    :return:
    """

    return 1 / (1 + np.exp(-x))


def bin_array(N, dtype=int):
    return (np.arange(1<<N, dtype=dtype)[:, None] >> np.arange(N, dtype=dtype)[::-1]) & 0b1


def exp_smoothing(x, s, alpha=0.5):

    return alpha * x + (1 - alpha) * s


def get_batches(X, n_batch, rng):
    """
    This returns an empty array at the end if the final index exceeds the array size.
    splits data into batches along first dimension
    :param X: data array, N x d
    :param n_batch: batch size
    :return:
    """

    N = X.shape[0]
    X_shuffled = rng.permutation(X)
    split_index = np.arange(0, N - N % n_batch + 1, n_batch)[1:]

    list = np.split(X_shuffled, split_index)
    if list[-1].shape[0] == 0:
        del list[-1]

    return list


def gaussian_initialize(size, rng, mu=0, sig=0.01):

    return rng.normal(mu, sig, size=size)


def add_regularization(theta, grad, weight_decay=None, sparsity=None):

    if weight_decay is not None and sparsity is not None:
        return grad + weight_decay * theta + sparsity * (theta != 0)
    elif weight_decay is not None:
        return grad + weight_decay * theta
    elif sparsity is not None:
        return grad + sparsity * (theta != 0)
    else:
        return grad


def grad_phase(phase_indicator, b):

    return ((-1) ** (1 + phase_indicator)) * (b ** -phase_indicator) * (1 - b) ** -(1 - phase_indicator)


def quad_energy(v, h, whv, whh, bh):

    if whh is None:
        return - h.T @ whv @ v - 1 / 2 * h.T @ bh
    else:
        return - h.T @ whv @ v - 1 / 2 * (h.T @ whh @ h + h.T @ bh)


def all_probs(Whv, b_h, Whh=None, n_hidden=None, beta=1):

    if Whh is None and n_hidden is None:
        print('error, must given either Whh or n_hidden')
        return None
    elif n_hidden is None:
        n_hidden = Whh.shape[0]
    n_total = Whv.shape[-1] + n_hidden

    support = bin_array(n_total)
    p_hats = []
    for x in support:
        p_hats.append(np.exp(- beta * quad_energy(x[:-n_hidden], x[-n_hidden:], Whv, Whh, b_h)))
    p_hats = np.asarray(p_hats)
    probs = p_hats / p_hats.sum()

    return support, probs


def get_marginals(s, p, n_visible):

    marginals = []
    marginal_support = bin_array(n_visible)
    for x in marginal_support:
        marginals.append(p[np.all(s[:, :n_visible] == x, axis=1)].sum())

    return marginal_support, np.asarray(marginals)


class SBM:

    def __init__(self, n_visible, n_hidden, rng, beta=1.0, n_outputs=10, K=1, model_type='SBM', seed=None):

        self.rng = rng
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.Whv = np.zeros((n_hidden, n_visible))
        self.Whh = np.zeros((n_hidden, n_hidden))
        self.b_h = np.zeros(n_hidden)
        self.beta = beta
        self.history = []
        self.v = np.zeros(n_visible)
        self.h = np.zeros(n_hidden)
        self.K = K
        self.model_type=model_type
        self.train_type = None
        self.train_learning_rate = None
        self.train_b = None
        self.seed = seed
        self.train_n_wake = []
        self.train_total_iters = None
        self.train_end_error = None

        if self.model_type == 'RBM':
            self.Whh = None

    def init_weights(self, mu, sig):

        self.Whv = gaussian_initialize((self.n_hidden, self.n_visible), rng=self.rng, mu=mu, sig=sig)
        if self.model_type == 'SBM':
            self.Whh = np.triu(gaussian_initialize((self.n_hidden, self.n_hidden), rng=self.rng, mu=mu, sig=sig), 1)
            self.Whh = self.Whh + self.Whh.T

    def init_bias(self, mu, sig):

        self.b_h = gaussian_initialize(self.n_hidden, rng=self.rng, mu=mu, sig=sig)
    
    def prob_hidden_units_rbm(self, v):

        return sigmoid(self.beta * (self.Whv @ v + self.b_h))
    
    def prob_hidden_unit(self, v, h, ix):
        
        return sigmoid(self.beta * (self.Whv[ix, :] @ v + self.Whh[ix, : ] @ h + self.b_h[ix]))

    def prob_visible_units(self, h, ixs, x=None, b=None):
        
        if x is None:
            return sigmoid(self.beta * (self.Whv[:, ixs].T @ h))
        else:
            mask = np.ones(self.n_visible)
            mask[ixs] = 0
            return sigmoid(self.beta * (self.Whv.T @ h + b * mask * (2 * x - 1)))

    def sample_v(self, v, h, idxs=None, x=None, b=None):
        if idxs is not None and x is None:
            unif_sample = self.rng.uniform(size=len(idxs))
            v[idxs] = self.prob_visible_units(h, idxs) >= unif_sample
        elif x is not None:
            unif_sample = self.rng.uniform(size=self.n_visible)
            v = self.prob_visible_units(h, idxs, x, b) >= unif_sample

        return v

    def sample_h(self, v, h):
        unif_sample = self.rng.uniform(size=self.n_hidden)
        if self.model_type == 'SBM':
            for ix, unit in enumerate(unif_sample):
                h[ix] = self.prob_hidden_unit(v, h, ix) >= unit
        elif self.model_type == 'RBM':
            h = self.prob_hidden_units_rbm(v) >= unif_sample
        return h

    def sample_one_step(self, v, h, idxs=None, x=None, b=None):
        h = self.sample_h(v, h)
        v = self.sample_v(v, h, idxs, x, b)

        return v.astype(int), h.astype(int)
    
    def sample(self, K, v=None, h=None, clamp_inputs=True, clamp_outputs=True, keep_traj=False, x=None, b=None):

        idxs = np.arange(0, self.n_visible)

        if clamp_inputs and clamp_outputs:
            idxs = None
        elif clamp_inputs:
            idxs = idxs[-self.n_outputs:]
        elif clamp_outputs:
            idxs = idxs[:-self.n_outputs]
        
        if v is None and h is None:
            v, h = np.copy(self.v), np.copy(self.h)
        elif v is not None and h is None:
            h = np.zeros(self.n_hidden)
        elif h is not None and v is None:
            v = np.zeros(self.n_visible)
        
        if keep_traj:
            V, H = [v], [h]
            for _ in range(K):
                v, h = self.sample_one_step(np.copy(V[-1]), np.copy(H[-1]), idxs, x, b)
                V.append(v)
                H.append(h)
            return np.asarray(V).T, np.asarray(H).T
        else:
            for _ in range(K):
                v, h = self.sample_one_step(v, h, idxs, x, b) 
            return v, h
            
    def mc_estimate(self, X, samples, burn_in):

        estimates = []
        X = np.copy(X)
        X[:, -self.n_outputs:] = np.zeros((X.shape[0], self.n_outputs))
        for x in X:
            v, _ = self.sample(samples, v=x, clamp_inputs=True, clamp_outputs=False, keep_traj=True)
            estimates.append(np.mean(v[-self.n_outputs:, burn_in:], axis=-1))

        return np.asarray(estimates)

    def mc_classify(self, X, samples, burn_in):

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        estimates = self.mc_estimate(X, samples, burn_in)

        binary_estimates = np.zeros((X.shape[0], self.n_outputs))
        max_ind = np.argmax(estimates, axis=-1)

        for ix, ind in enumerate(max_ind):
            binary_estimates[ix, ind] = 1

        return binary_estimates
    
    def mnist_error_1(self, X, samples, burn_in):

        estimates = self.mc_classify(X, samples, burn_in)
        true = X[:, -self.n_outputs:]

        return np.sum(np.any(estimates != true, axis=-1)) / estimates.shape[0]
    
    def calc_h_prior_unnormalized(self, H):

        h_prior = np.zeros(H.shape[0])
        for ind, h in enumerate(H):
            h_prior[ind] = np.exp(self.beta * np.sum(h * self.b_h))

        return h_prior

    def log_partition(self, H, h_prior_unnormalized):

        Z = 0
        for ind, h in enumerate(H):
            v_prod = np.prod(1 + np.exp(self.beta * (h @ self.Whv)))
            Z += h_prior_unnormalized[ind] * v_prod
        
        return np.log(Z)

    def lp_point_unnormalized(self, x, H, h_prior_unnormalized):

        p_unnormalized = 0
        for ind, h in enumerate(H):
            v_prod = np.prod(np.exp(self.beta * x * (h @ self.Whv)))
            p_unnormalized += h_prior_unnormalized[ind] * v_prod

        return np.log(p_unnormalized)
        
    def log_likelihood_1(self, X):

        H = bin_array(self.n_hidden)
        hpu = self.calc_h_prior_unnormalized(H)
        lp = self.log_partition(H, hpu)
        ll_unnormalized = 0
        for x in X:
            ll_unnormalized += self.lp_point_unnormalized(x, H, hpu)
        
        return - ll_unnormalized + lp * X.shape[0]
    
    def total_variation(self, p_star):

        support, probs = all_probs(self.Whv, self.b_h, self.Whh, self.n_hidden, beta=self.beta)
        _, probs = get_marginals(support, probs, n_visible=self.n_visible)

        return 1 / 2 * np.sum(np.abs(probs - p_star))
    
    def tv_and_ll(self, X, p_star):

        support, probs = all_probs(self.Whv, self.b_h, self.Whh, self.n_hidden, beta=self.beta)
        marginal_support, probs = get_marginals(support, probs, n_visible=self.n_visible)

        ll = 0
        for x in X:
            ll += - np.log(probs[np.all(marginal_support == x, axis=1)].sum())

        return ll, 1 / 2 * np.sum(np.abs(probs - p_star))
    
    def get_history(self, X, samples=500, burn_in=100, train_error_samples=100, p_star=None, type='mnist_error'):

        if type == 'mnist_error':
            return self.mnist_error_1(self.rng.permutation(X)[:train_error_samples, :], samples, burn_in)
        elif type == 'nll':
            return self.log_likelihood_1(X)
        elif type == 'total_variation':
            return self.total_variation(p_star)
        elif type == 'tv_and_ll':
            return self.tv_and_ll(X, p_star)
    
    def add_history(self, X, samples, burn_in, train_error_samples, idx, wandb, wpc=None, p_star=None, type='mnist_error'):

        self.history.append(self.get_history(X, samples=samples, burn_in=burn_in, train_error_samples=train_error_samples, p_star=p_star, type=type))

        if wandb:
            wb.log({'iteration': idx, type: self.history[-1], 'wake periods': wpc})  # log for wandb
        if wpc is not None:
            print(f'training -- iteration: {idx}, {type}: {self.history[-1]}, wake periods: {wpc}')
            self.train_n_wake.append(wpc)
        else:
            print(f'training -- iteration: {idx}, {type}: {self.history[-1]}')

    def cdk_update(self, batch, learning_rate=0.01, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None):

        dWhv = 0
        db_h = 0
        if self.model_type == 'SBM':
            dWhh = 0

        for s in batch:

            self.v, self.h = s, np.zeros(self.n_hidden)
            v_pos, h_pos = self.sample(self.K, clamp_inputs=True, clamp_outputs=True, keep_traj=False)
            self.v, self.h = s, np.zeros(self.n_hidden)
            v_neg, h_neg = self.sample(self.K, clamp_inputs=False, clamp_outputs=False, keep_traj=False)

            dWhv += add_regularization(self.Whv, np.outer(h_neg, v_neg) - np.outer(h_pos, v_pos), wd_hv, sp_hv) 
            db_h += add_regularization(self.b_h, h_neg - h_pos, wd_bias, sp_bias)
            if self.model_type == 'SBM':
                dWhh += add_regularization(self.Whh, np.outer(h_neg, h_neg) - np.outer(h_pos, h_pos), wd_hh, sp_hh)

        grad_scaling = learning_rate * self.beta
        self.Whv += - grad_scaling * dWhv
        self.b_h += - grad_scaling * db_h
        if self.model_type == 'SBM':
            self.Whh += - grad_scaling * dWhh * (np.eye(len(self.Whh)) == 0)

    def train_cdk(self, X, n_epochs, n_batch=1, learning_rate=0.01, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None, samples=500, burn_in=100, train_error_samples=100, history_step_size=1, p_star=None, wandb=False, max_time_mins=None, history_type='mnist_error'):

        self.train_type = 'cdk'
        self.train_learning_rate = learning_rate
        t0 = time.time()

        iter = 0
        for _ in range(n_epochs):
            #print(f'Training: Epoch {epoch + 1}...')
            batches = get_batches(X, n_batch, self.rng)

            for batch in batches:

                if iter % history_step_size == 0:
                    self.add_history(X, samples, burn_in, train_error_samples, iter, wandb, p_star=p_star, type=history_type)

                self.cdk_update(batch, learning_rate=learning_rate, wd_hv=wd_hv, sp_hv=sp_hv, wd_bias=wd_bias, sp_bias=sp_bias, wd_hh=wd_hh, sp_hh=sp_hh)
                iter += 1

                if max_time_mins:
                    if (time.time() - t0) / 60 >= max_time_mins:
                        break

            if max_time_mins:
                if (time.time() - t0) / 60 >= max_time_mins:
                    print('training end triggered by time limit')
                    break
            
        self.add_history(X, samples, burn_in, train_error_samples, iter + 1, wandb, p_star=p_star, type=history_type)
        print('training finished!')

    def npl_update(self, learning_rate, phase_indicator, b=0.5, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None):

        for _ in range(self.K):
            
            if phase_indicator:
                self.v, self.h = self.sample(1, clamp_inputs=True, clamp_outputs=True, keep_traj=False)
            else:
                self.v, self.h = self.sample(1, clamp_inputs=False, clamp_outputs=False, keep_traj=False)

            grad_scaling = learning_rate * self.beta
            phase_scaling = grad_phase(phase_indicator, b)
            self.Whv += - grad_scaling * add_regularization(self.Whv, - np.outer(self.h, self.v) * phase_scaling, wd_hv, sp_hv)
            self.b_h += - grad_scaling * add_regularization(self.b_h, - self.h * phase_scaling, wd_bias, sp_bias)
            if self.model_type == 'SBM':
                self.Whh += - grad_scaling * add_regularization(self.Whh, - np.outer(self.h, self.h) * phase_scaling, wd_hh, sp_hh) * (np.eye(len(self.Whh)) == 0)

    def train_npl(self, X, n_iters, learning_rate=0.01, b=0.5, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None, samples=500, burn_in=100, train_error_samples=100, history_step_size=1, history_type='mnist_error', p_star=None, wandb=False, max_time_mins=None):

        self.train_type = 'npl'
        self.train_learning_rate = learning_rate * self.K
        self.train_b = b
        t0 = time.time()
        data_dim = X.shape[0]

        wake_phase_count, iter = 0, 0
        while iter < n_iters:

            phase_indicator = int(self.rng.uniform() <= b)
            if phase_indicator:
                self.v = X[wake_phase_count % data_dim, :]
            
            if iter % history_step_size == 0:
                self.add_history(X, samples, burn_in, train_error_samples, iter, wandb, wake_phase_count, p_star=p_star, type=history_type)

            self.npl_update(learning_rate, phase_indicator, b, wd_hv=wd_hv, sp_hv=sp_hv, wd_bias=wd_bias, sp_bias=sp_bias, wd_hh=wd_hh, sp_hh=sp_hh)
            wake_phase_count += phase_indicator
            iter += 1
            
            if max_time_mins:
                if (time.time() - t0) / 60 >= max_time_mins:
                    print('training end triggered by time limit')
                    break
            
        self.add_history(X, samples, burn_in, train_error_samples, iter + 1, wandb, wake_phase_count, p_star=p_star, type=history_type)
        self.train_total_iters = iter + 1
        print('training finished!')
    
    def npl_offline_update(self, batch, b=0.5, learning_rate=0.01, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None):

        dWhv = 0
        db_h = 0
        if self.model_type == 'SBM':
            dWhh = 0

        wake_phase_count = 0
        while wake_phase_count < len(batch):

            phase_indicator = int(self.rng.uniform() <= b)
            if phase_indicator:
                self.v = batch[wake_phase_count, :]
                wake_phase_count += 1
                self.v, self.h = self.sample(self.K, clamp_inputs=True, clamp_outputs=True, keep_traj=False)
            else:
                self.v, self.h = self.sample(self.K, clamp_inputs=False, clamp_outputs=False, keep_traj=False)

            phase_scaling = grad_phase(phase_indicator, b)
            dWhv += add_regularization(self.Whv, - np.outer(self.h, self.v) * phase_scaling, wd_hv, sp_hv)
            db_h += self.h * add_regularization(self.b_h, - self.h * phase_scaling, wd_bias, sp_bias)
            if self.model_type == 'SBM':
                dWhh += add_regularization(self.Whh, - np.outer(self.h, self.h) * phase_scaling, wd_hh, sp_hh)

        grad_scaling = learning_rate * self.beta
        self.Whv += - grad_scaling * dWhv
        self.b_h += - grad_scaling * db_h
        if self.model_type == 'SBM':
            self.Whh += - grad_scaling * dWhh * (np.eye(len(self.Whh)) == 0)
    
    def train_npl_offline(self, X, n_epochs, n_batch=1, learning_rate=0.01, b=0.5, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None, samples=500, burn_in=100, train_error_samples=100, history_step_size=1, wandb=False, max_time_mins=None, p_star=None, history_type='mnist_error'):
        """
        Note that this is the same as train_cdk except has a different update function (see <<below lines>>)
        """

        self.train_type = 'isd'
        self.train_learning_rate = learning_rate
        self.train_b = b
        t0 = time.time()

        iter = 0
        for epoch in range(n_epochs):
            print(f'Training: Epoch {epoch + 1}...')
            batches = get_batches(X, n_batch, self.rng)

            for batch in batches:

                if iter % history_step_size == 0:
                    self.add_history(X, samples, burn_in, train_error_samples, iter, wandb, p_star=p_star, type=history_type)

                self.npl_offline_update(batch, b=b, learning_rate=learning_rate, wd_hv=wd_hv, sp_hv=sp_hv, wd_bias=wd_bias, sp_bias=sp_bias, wd_hh=wd_hh, sp_hh=sp_hh) #<<below lines>>
                iter += 1

                if max_time_mins:
                    if (time.time() - t0) / 60 >= max_time_mins:
                        break
            
            if max_time_mins:
                if (time.time() - t0) / 60 >= max_time_mins:
                    print('training end triggered by time limit')
                    break

        self.add_history(X, samples, burn_in, train_error_samples, iter + 1, wandb, p_star=p_star, type=history_type)
        print('training finished!')
    
    def nplr_update(self, learning_rate, phase_indicator, b=0.5, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None):

        if phase_indicator:
            self.v, self.h = self.sample(1, clamp_inputs=True, clamp_outputs=True, keep_traj=False)
        else:
            self.v, self.h = self.sample(1, clamp_inputs=False, clamp_outputs=False, keep_traj=False)

        grad_scaling = learning_rate * self.beta
        phase_scaling = grad_phase(phase_indicator, b)
        self.Whv += - grad_scaling * add_regularization(self.Whv, - np.outer(self.h, self.v) * phase_scaling, wd_hv, sp_hv)
        self.b_h += - grad_scaling * add_regularization(self.b_h, - self.h * phase_scaling, wd_bias, sp_bias)
        if self.model_type == 'SBM':
            self.Whh += - grad_scaling * add_regularization(self.Whh, - np.outer(self.h, self.h) * phase_scaling, wd_hh, sp_hh) * (np.eye(len(self.Whh)) == 0)

    def train_nplr(self, X, n_iters, learning_rate=0.01, b=0.5, wd_hv=None, sp_hv=None, wd_bias=None, sp_bias=None, wd_hh=None, sp_hh=None, samples=500, burn_in=100, train_error_samples=100, history_step_size=1, history_type='mnist_error', p_star=None, wandb=False, max_time_mins=None, p_phase_switch=None):

        self.train_type = 'npl'
        self.train_learning_rate = learning_rate * self.K
        self.train_b = b
        t0 = time.time()
        data_dim = X.shape[0]
        phase_indicator = int(self.rng.uniform() <= b)
        if phase_indicator:
            self.v = X[0, :]
        
        wake_phase_count, iter = 0, 0
        self.add_history(X, samples, burn_in, train_error_samples, iter, wandb, wake_phase_count, p_star=p_star, type=history_type)
        while iter < n_iters:
            
            self.nplr_update(learning_rate, phase_indicator, b, wd_hv=wd_hv, sp_hv=sp_hv, wd_bias=wd_bias, sp_bias=sp_bias, wd_hh=wd_hh, sp_hh=sp_hh)

            period_switch = int(self.rng.uniform() <= p_phase_switch)
            if period_switch:
                iter += 1
                phase_indicator = int(self.rng.uniform() <= b)
                if phase_indicator:
                    wake_phase_count += phase_indicator
                    self.v = X[wake_phase_count % data_dim, :]
                if iter % history_step_size == 0:
                    self.add_history(X, samples, burn_in, train_error_samples, iter, wandb, wake_phase_count, p_star=p_star, type=history_type)

            if max_time_mins:
                if (time.time() - t0) / 60 >= max_time_mins:
                    print('training end triggered by time limit')
                    break
            
        print('training finished!')

    def gen_class(self, class_num, samples, burn_in):

        given = np.zeros(self.n_outputs)
        given[class_num] = 1
        v = np.concatenate((np.zeros(self.n_visible - self.n_outputs), given))

        V, _ = self.sample(samples + burn_in, v=v, clamp_inputs=False, clamp_outputs=True, keep_traj=True)

        return V[:-self.n_outputs, burn_in + 1:]

    def class_samples(self, burn_in, mean_over=1):

        images = []
        for num in range(self.n_outputs):
            images.append(np.mean(self.gen_class(num, mean_over, burn_in), axis=-1))

        return np.asarray(images)
        

#%%

if __name__ == "__main__":

    rng = Generator(PCG64(None))

    #%% LOAD DATA
    dataset = 'MNIST'
    if dataset == 'BAS':
        with open('BAS_42.pkl', 'rb') as f:
            load_dict = pic.load(f)
        data = load_dict['train_data_vector']
        n_outputs = 0
    elif dataset == 'MNIST':
        data, test_data = mt.mnist_for_gen(index_list=None, return_torch=False, thresh=0.8)
        data = rng.permutation(data)
        n_outputs = 10

    #%% DEFINE HYPER-PARAMS
    n_v = data.shape[-1]
    n_h = 16
    alpha = 0.005  # learning rate
    
    mu_weight_prior = 0 # parameter initialization
    sig_weight_prior = 0.01

    K = 200 # time steps per phase
    alpha_scale = 1 / K
    mod_ty = 'RBM' # model type: RBM architechture selected
    b = 0.5  # positive phase probability

    num_epochs = 1
    batch_size = 1
    n_iters = 80000

    #%% INITIALIZE RBM
    model = SBM(K=K, n_visible=n_v, n_hidden=n_h, rng=rng, beta=1, n_outputs=n_outputs, model_type=mod_ty)
    model.init_weights(mu=mu_weight_prior, sig=sig_weight_prior)
    
    # PARAMETERS FOR MONTE CARLO CLASSIFICATION ERROR CALCULATION (USED FOR MNIST ONLY)
    bench_samples = 80  # length of Monte Carlo chains to run for each sample
    bench_burn_in = 40  # burn in for Monte Carlo chains
    error_samples = 20  # total number of samples run for each prediction

    history_step_size = 100  # how many phases are run between consecutive error calculations during training

    #%% TRAIN MODEL
    model.train_cdk(data, num_epochs, batch_size, learning_rate=alpha, samples=bench_samples, burn_in=bench_burn_in, train_error_samples=error_samples, history_step_size=history_step_size)
    model.train_npl(data, n_iters, learning_rate=alpha * alpha_scale, b=b, samples=bench_samples, burn_in=bench_burn_in, train_error_samples=error_samples, history_step_size=history_step_size)
    #model.train_npl_offline(data, num_epochs, n_batch=batch_size, learning_rate=alpha, samples=bench_samples, burn_in=bench_burn_in, train_error_samples=error_samples, history_step_size=history_step_size)

    #%% CALCULATE END OF TRAINING ERROR FOR MNIST
    train_error = model.mnist_error_1(data[:1000, :], samples=200, burn_in=100)
    print(f'full training error: {train_error}')

    #test_error = model.mnist_error_1(test_data, samples=300, burn_in=200)
    #print(f'full testing error: {test_error}')


# %%
