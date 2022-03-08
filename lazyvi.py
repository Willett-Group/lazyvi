import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import copy
import seaborn as sns
sns.set(style="ticks", font_scale=1.2)
from matplotlib import pyplot as plt

from utils import *


"""
lazy training
"""

class LazyVI:
    def __init__(self, full_nn, cv=5, lambda_path=np.logspace(0, 2, 10), draw_path=False, lam=0):
        self.full_nn = full_nn
        self.lambda_path = lambda_path
        self.cv = cv
        self.cv_results = None
        self.lam = lam
        self.draw_path = draw_path

    def flat_tensors(self, T_list):
        """
        Flatten a list of tensors to a vector, and store the original shapes of the tensors for future recovery
        output: Tuple[tensor, list]
        """
        tensor_info = [t.shape for t in T_list]
        flattened_tensors = torch.cat([t.reshape(-1) for t in T_list])
        return flattened_tensors, tensor_info

    def recover_tensors(self, T: torch.Tensor):
        """
        recover the parameter tensors in order to feed into a neural network
        output: a list of tensors
        """
        i = 0
        res = []
        for s in self.shape_info:
            len_s = np.prod(s)
            res.append(T[i:i+len_s].reshape(s))
            i += len_s
        return res

    def extract_grad(self, X):
        """
        extract gradients from trained network
        output: n x (# network params) matrix
        """
        grads = []
        n = X.shape[0]
        params_full = tuple(self.full_nn.parameters())
        flat_params, shape_info = self.flat_tensors(params_full)
        for i in range(n):
            # calculate the first order gradient wrt all parameters
            if len(X.shape) > 2:
                yi = self.full_nn(X[[i]])
            else:
                yi = self.full_nn(X[i])
            this_grad = torch.autograd.grad(yi, params_full, create_graph=True)
            flat_this_grad, _ = self.flat_tensors(this_grad)
            grads.append(flat_this_grad)
        grads = np.array([grad.detach().numpy() for grad in grads])
        self.grads = grads
        self.flat_params = flat_params
        self.shape_info = shape_info
        return self

    def lazy_predict(self, X, y, grads, lam):
        #_, p = X.shape
        dr_pred_train = self.full_nn(X)
        lazy = Ridge(alpha=lam).fit(grads, y - dr_pred_train.detach().numpy())
        delta = lazy.coef_
        lazy_retrain_params = torch.FloatTensor(delta) + self.flat_params
        lazy_retrain_Tlist = self.recover_tensors(lazy_retrain_params.reshape(-1))
        # create deep copy of full_nn
        lazy_retrain_nn = copy.deepcopy(self.full_nn)
        # update parameters with lazy correction
        for k, param in enumerate(lazy_retrain_nn.parameters()):
            param.data = lazy_retrain_Tlist[k]
        self.lazy_retrain_nn = lazy_retrain_nn
        return self

    def fit(self, X, y):
        kf = KFold(n_splits=self.cv, shuffle=True)
        errors = []
        self.extract_grad(X)
        # cross validate for ridge parameter
        if self.lam == 0:
            for lam in self.lambda_path:
                for train, test in kf.split(X):
                    self.lazy_predict(X[train], y[train], self.grads[train], lam)
                    lazy_pred_test = self.lazy_retrain_nn(X[test])
                    errors.append([lam, nn.MSELoss()(lazy_pred_test, y[test]).item()])
            errors = pd.DataFrame(errors, columns=['lam', 'mse'])
            lam = errors.groupby(['lam']).mse.mean().sort_values().index[0]
            self.cv_results = errors.groupby(['lam']).mse.mean().sort_index().values
            self.lam = lam
        self.lazy_predict(X, y, self.grads, self.lam)
        if self.draw_path:
            sns.lineplot(x='lam', y='mse', data=errors)
            plt.title(f'$\lambda^* = {self.lam.round(2)}$')

    def predict(self, X):
        return self.lazy_retrain_nn(X)
