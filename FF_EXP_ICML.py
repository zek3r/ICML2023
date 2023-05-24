#%%
import torch as tc
import MNIST_TOOLS_ICML as mt
import torch.nn as nn
from torch.optim import Adam


def get_batches(X, n_batch):
    """
    This returns an empty array at the end if the final index exceeds the array size.
    splits data into batches along first dimension
    :param X: data array, N x d
    :param n_batch: batch size
    :return:
    """

    inx = tc.randperm(X.shape[0])
    X_shuffled = X[inx, :]
    array = tc.split(X_shuffled, n_batch)

    return array


def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(nn.Module):
    """
    Feed-forward ANN
    """

    def __init__(self, dims, lr):
        """
        :param dims: list of layer dimensions, from input through output
        """
        super().__init__()
        self.history = []
        self.test_history = []
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], lr)]
    
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = tc.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)
    
    def train(self, x_pos, x_neg, x, y, x_te, y_te, num_epochs=None, batch_size=None, b=None, history_step=1000):
        for i in range(num_epochs):
            pos_batches = get_batches(x_pos, batch_size)
            neg_batches = get_batches(x_neg, batch_size)
            print(f'training epoch: {i}')
            n = len(pos_batches)
            if b is None:
                for j in range(n):
                    self.rec_history(x, y, x_te, y_te, j, history_step)
                    h_pos, h_neg = pos_batches[j], neg_batches[j]
                    for layer in self.layers:
                        h_pos, h_neg = layer.train(h_pos, h_neg)
            else:
                pos_count, neg_count = 0, 0
                for j in range(n):
                    self.rec_history(x, y, x_te, y_te, j, history_step)
                    B = tc.bernoulli(tc.Tensor([b]))
                    if B:
                        h = pos_batches[pos_count]
                        for layer in self.layers:
                            h = layer.train_isd(h, B=B)
                        pos_count += 1
                    else:
                        h = neg_batches[neg_count]
                        for layer in self.layers:
                            h = layer.train_isd(h, B=B)
                        neg_count += 1
            self.rec_history(x, y, x_te, y_te, n, history_step, iterating=False)
    
    def rec_history(self, x, y, x_te, y_te, inx, step, iterating=True):
        if inx % step == 0:
            error_train = 1.0 - self.predict(x).eq(y).float().mean().item()
            error_test = 1.0 - self.predict(x_te).eq(y_te).float().mean().item()
            self.history.append(error_train)
            self.test_history.append(error_test)
            if iterating:
                print(f'training iter: {inx}; error: {error_train}', end='\r')
            else:
                print(f'training iter: {inx}; error: {error_train}')


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, lr=0.03,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = tc.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=lr)
        self.threshold = 2.0

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            tc.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):

        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)
        g_pos = h_pos.pow(2).mean(1)
        g_neg = h_neg.pow(2).mean(1)

        loss = tc.log(1 + tc.exp(tc.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()
        self.opt.zero_grad()
        # this backward just compute the derivative and is thus not doing backpropagation
        loss.backward()
        self.opt.step()
        return h_pos.detach(), h_neg.detach()
    
    def train_isd(self, x, B):

        h = self.forward(x)
        g = h.pow(2).mean(1)

        if B:
            loss = 1 / B * tc.log(1 + tc.exp(-g + self.threshold))
        else:
            loss = 1 / (1 - B) * tc.log(1 + tc.exp(g - self.threshold))
        self.opt.zero_grad()
        # this backward just compute the derivative and is thus not doing backpropagation
        loss.backward()
        self.opt.step()
        return h.detach()


if __name__ == "__main__":
    tc.manual_seed(1234)
    rng = tc.Generator()
    rng.manual_seed(1234)

    #%% LOAD DATA
    use_numbers = None
    x, y, x_te, y_te = mt.mnist_for_class(index_list=use_numbers, return_torch=True, thresh=None) # probs need to normalize these
    y, y_te = y.to(tc.long), y_te.to(tc.long)
    x, x_te = x.to(tc.float32), x_te.to(tc.float32)
    x_no_label = x.clone()
    x_no_label[:, :10] *= 0.0

    #%% GENERATE NEGATIVE SAMPLES
    rnd = tc.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    #%% INITIALIZE STANDARD CL MODEL
    learning_rate = 0.01
    net0 = Net([784, 500, 500], lr=learning_rate)
    x_pos = overlay_y_on_x(x, y)

    #%% TRAIN STANDARD CL
    history_step_size = 1000
    n_epochs = 2
    n_batch = 1
    net0.train(x_pos, x_neg, x, y, x_te, y_te, num_epochs=n_epochs, batch_size=n_batch, history_step=history_step_size)

    print('train error:', 1.0 - net0.predict(x).eq(y).float().mean().item())

    #%% INITIALIZE ISD MODEL
    learning_rate = 0.01
    net = Net([784, 500, 500], lr=learning_rate)
    x_pos = overlay_y_on_x(x, y)

    #%% TRAIN ISD
    history_step_size = 1000
    n_epochs = 2
    n_batch = 1
    b = 0.5
    net.train(x_pos, x_neg, x, y, x_te, y_te, b=b, num_epochs=n_epochs, batch_size=n_batch, history_step=history_step_size)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    #%% PRINT TEST ERROR
    print('test error:', 1.0 - net0.predict(x_te).eq(y_te).float().mean().item())
    print('isd test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())


# %%
