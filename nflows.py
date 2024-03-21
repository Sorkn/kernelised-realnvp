import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import os


if DEVICE:=os.environ.get('device'):
    DEVICE = torch.device(DEVICE)
else:
    DEVICE = torch.device('cpu')


class InvertibleLayer(nn.Module):
    def __init__(self, var_size):
        super(InvertibleLayer, self).__init__()

        self.var_size = var_size


    def f(self, X, C):
        pass


    def g(self, X, C):
        pass



class NormalizingFlow(nn.Module):

    def __init__(self, layers, prior):
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior


    def log_prob(self, X, C):

        log_likelihood = None

        for layer in self.layers:
            X, change = layer.f(X, C)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(X)

        return log_likelihood.mean()


    def sample(self, C):
        if type(C) == type(1):
            n = C
            C = None
        else:
            n = len(C)

        X = self.prior.sample((n,))
        for layer in self.layers[::-1]:
            X = layer.g(X, C)

        return X