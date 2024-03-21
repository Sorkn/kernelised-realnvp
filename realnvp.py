import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import os
from tqdm.notebook import tqdm
from nflows import InvertibleLayer, NormalizingFlow
from sklearn.gaussian_process.kernels import RBF
import numpy as np


if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')


def gen_network(n_inputs, n_outputs, hidden=(10,), activation='tanh'):

    model = nn.Sequential()
    for i in range(len(hidden)):

        # add layer
        if i == 0:
            alayer = nn.Linear(n_inputs, hidden[i])
        else:
            alayer = nn.Linear(hidden[i-1], hidden[i])
        model.append(alayer)

        # add activation
        if activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            act = nn.ReLU()
        model.append(act)

    # output layer
    model.append(nn.Linear(hidden[-1], n_outputs))

    return model


class RealNVPLayer(InvertibleLayer):

    def __init__(self, var_size, cond_size, mask, hidden=(10,), activation='tanh', kernelized=False):
        super(RealNVPLayer, self).__init__(var_size=var_size)
        self.auxiliary_points = None
        self.mask = mask.to(DEVICE)
        self.kernelized = kernelized
        self.hidden = hidden
        if kernelized:
            # self.auxiliary_points = np.random.rand(hidden, var_size + cond_size)
            
            Wt = torch.randn(var_size + cond_size, hidden, dtype=torch.float32, requires_grad=True)
            Ws = torch.randn(var_size + cond_size, hidden, dtype=torch.float32, requires_grad=True)
            self.K = RBF(1.0)
            
            self.Wt = nn.Parameter(Wt)
            self.Ws = nn.Parameter(Ws)

        else:
            self.nn_t = gen_network(var_size + cond_size, var_size, hidden, activation)
            self.nn_s = gen_network(var_size + cond_size, var_size, hidden, activation)


    def f(self, X, C=None):
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = (X * self.mask[None, :])
        if self.auxiliary_points is None:
            self.auxiliary_points = XC[np.random.choice(len(X), self.hidden)].detach().numpy()
        if self.kernelized:
            kernel_l = torch.transpose(torch.from_numpy(self.K(XC.detach().numpy(), self.auxiliary_points)), 0, 1).float()
            T = torch.transpose(self.Wt @ kernel_l, 0, 1)
            S = torch.transpose(self.Ws @ kernel_l, 0, 1)
            if C is not None:
                T = T[:, :-1]
                S = S[:, :-1]
        else:
            T = self.nn_t(XC)
            S = self.nn_s(XC)

        X_new = (X * torch.exp(S) + T) * (1 - self.mask[None, :]) + X * self.mask[None, :]
        log_det = (S * (1 - self.mask[None, :])).sum(dim=-1)
        return X_new, log_det


    def g(self, X, C=None):
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]
        if self.kernelized:
            kernel_l = torch.transpose(torch.from_numpy(self.K(XC.detach().numpy(), self.auxiliary_points)), 0, 1).float()
            T = torch.transpose(self.Wt @ kernel_l, 0, 1)
            S = torch.transpose(self.Ws @ kernel_l, 0, 1)
            if C is not None:
                T = T[:, :-1]
                S = S[:, :-1]
        else:
            T = self.nn_t(XC)
            S = self.nn_s(XC)

        X_new = ((X - T) * torch.exp(-S)) * (1 - self.mask[None, :]) + X * self.mask[None, :]
        return X_new



class RealNVP(object):

    def __init__(self, n_layers=8, hidden=(10,), activation='tanh',
                       batch_size=32, n_epochs=10, lr=0.0001, weight_decay=0, verbose=0, kernelized=False):
        super(RealNVP, self).__init__()

        self.n_layers = n_layers
        self.hidden = hidden
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.kernelized = kernelized
        self.prior = None
        self.nf = None
        self.opt = None

        self.loss_history = []


    def _model_init(self, X, C):

        var_size = X.shape[1]
        if C is not None:
            cond_size = C.shape[1]
        else:
            cond_size = 0

        # init prior
        if self.prior is None:
            self.prior = torch.distributions.MultivariateNormal(torch.zeros(var_size, device=DEVICE),
                                                                torch.eye(var_size, device=DEVICE))
        # init NF model and optimizer
        if self.nf is None:

            layers = []
            for i in range(self.n_layers):
                alayer = RealNVPLayer(var_size=var_size,
                                      cond_size=cond_size,
                                      mask=((torch.arange(var_size) + i) % 2),
                                      hidden=self.hidden,
                                      activation=self.activation,
                                      kernelized=self.kernelized)
                layers.append(alayer)

            self.nf = NormalizingFlow(layers=layers, prior=self.prior).to(DEVICE)
            self.opt = torch.optim.Adam(self.nf.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)


    def fit(self, X, C=None):

        # model init
        self._model_init(X, C)

        # numpy to tensor, tensor to dataset
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        if C is not None:
            C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
            dataset = TensorDataset(X, C)
            C.to(DEVICE)
        else:
            dataset = TensorDataset(X)
        X.to(DEVICE)

        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=100, gamma=0.5)
        _range = range(self.n_epochs) if self.verbose<1 else tqdm(range(self.n_epochs), unit='epoch')
        for epoch in _range:
            if self.kernelized:
                # caiculate loss  
                if C is not None:
                    loss = -self.nf.log_prob(X, C)
                else:
                    loss = -self.nf.log_prob(X)
                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                scheduler.step()

                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())

                if self.verbose >= 2:
                    display_delta = max(1, X.shape[0] // self.verbose)
                    if i % display_delta == 0:
                        _range.set_description(f"loss: {loss:.4f}")
            else:
                for i, batch in enumerate(DataLoader(dataset, batch_size=self.batch_size, shuffle=True)):

                    X_batch = batch[0].to(DEVICE)
                    if C is not None:
                        C_batch = batch[1].to(DEVICE)
                    else:
                        C_batch = None

                    # caiculate loss
                    loss = -self.nf.log_prob(X_batch, C_batch)

                    # optimization step
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    # caiculate and store loss
                    self.loss_history.append(loss.detach().cpu())

                    if self.verbose >= 2:
                        display_delta = max(1, (X.shape[0] // self.batch_size) // self.verbose)
                        if i % display_delta == 0:
                            _range.set_description(f"loss: {loss:.4f}")
            
            if self.verbose == 1:
                _range.set_description(f"loss: {loss:.4f}")


    def sample(self, C=100):

        if type(C) != type(1):
            C = torch.tensor(C, dtype=torch.float32, device=DEVICE)
        X = self.nf.sample(C).cpu().detach().numpy()
        return X