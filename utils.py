import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
import scipy

"""
Network 
"""
class NN4vi(pl.LightningModule):

    def __init__(self,
                 input_dim: int,
                 hidden_widths: list,
                 output_dim: int,
                 activation = nn.ReLU):
        super().__init__()
        structure = [input_dim] + list(hidden_widths) + [output_dim]
        layers = []
        for j in range(len(structure) - 1):
            act = activation if j < len(structure) - 2 else nn.Identity
            layers += [nn.Linear(structure[j], structure[j + 1]), act()]

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        y_hat = self.net(x)
        loss = nn.MSELoss()(y_hat, y)
        # Logging to TensorBoard by default
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.MSELoss()(self.net(x), y)
        self.log('val_loss', loss)
        return loss


"""
VI helpers
"""

def dropout(X, i):
    X = np.array(X)
    N = X.shape[0]
    X_change = np.copy(X)
    X_change[:, i] = np.ones(N) * np.mean(X[:, i])
    X_change = torch.tensor(X_change, dtype=torch.float32)
    return X_change

def retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, exp_name, tol=1e-5):
    retrain_nn = NN4vi(p, hidden_layers, 1)
    #tb_logger = pl.loggers.TensorBoardLogger('logs/{}'.format(exp_name), name='retrain_{}'.format(j))
    early_stopping = EarlyStopping('val_loss', min_delta=tol)

    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train_change, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(trainset, batch_size=256)

    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=10000)
    trainer.fit(retrain_nn, train_loader, train_loader)

    retrain_pred_train = retrain_nn(X_train_change)
    retrain_pred_test = retrain_nn(X_test_change)
    return retrain_pred_train, retrain_pred_test


"""
lazy training
"""

def flat_tensors(T_list: list):
    """
    Flatten a list of tensors to a vector, and store the original shapes of the tensors for future recovery
    output: Tuple[tensor, list]
    """
    info = [t.shape for t in T_list]
    res = torch.cat([t.reshape(-1) for t in T_list])
    return res, info

def recover_tensors(T: torch.Tensor, info: list):
    """
    recover the parameter tensors in order to feed into a neural network
    output: a list of tensors
    """
    i = 0
    res = []
    for s in info:
        len_s = np.prod(s)
        res.append(T[i:i+len_s].reshape(s))
        i += len_s
    return res

def extract_grad(X, full_nn):
    """
    extract gradients from trained network
    output: n x (# network params) matrix
    """
    grads = []
    n, p = X.shape
    params_full = tuple(full_nn.parameters())
    flat_params, shape_info = flat_tensors(params_full)
    for i in range(n):
        # calculate the first order gradient wrt all parameters
        yi = full_nn(X[i])
        this_grad = torch.autograd.grad(yi, params_full, create_graph=True)
        flat_this_grad, _ = flat_tensors(this_grad)
        grads.append(flat_this_grad)
    grads = np.array([grad.detach().numpy() for grad in grads])
    return grads, flat_params, shape_info


def lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info, X_train, y_train, X_test, lam):
    _, p = X_train.shape
    dr_pred_train = full_nn(X_train)
    lazy = Ridge(alpha=lam).fit(grads, y_train - dr_pred_train.detach().numpy())
    delta = lazy.coef_
    lazy_retrain_params = torch.FloatTensor(delta) + flat_params
    lazy_retrain_Tlist = recover_tensors(lazy_retrain_params.reshape(-1), shape_info)
    lazy_retrain_nn = NN4vi(p, hidden_layers, 1)
    # hidden_layers probably doesn't need to be an argument here - get it from the structure
    for k, param in enumerate(lazy_retrain_nn.parameters()):
        param.data = lazy_retrain_Tlist[k]
    lazy_pred_train = lazy_retrain_nn(X_train)
    lazy_pred_test = lazy_retrain_nn(X_test)

    return lazy_pred_train, lazy_pred_test


def lazy_train_cv(full_nn, X_train_change, X_test_change, y_train, hidden_layers,
                  lam_path=np.logspace(-3, 3, 20), file=False):
    kf = KFold(n_splits=5, shuffle=True)
    errors = []
    grads, flat_params, shape_info = extract_grad(X_train_change, full_nn)
    print(grads.shape)

    for lam in lam_path:
        for train, test in kf.split(X_train_change):
            dr_pred_train = full_nn(X_train_change[train])
            grads_train = grads[train]
            lazy_pred_train, lazy_pred_test = lazy_predict(grads_train, flat_params, full_nn, hidden_layers, shape_info,
                                                           X_train_change[train], y_train[train], X_train_change[test],
                                                           lam)
            errors.append([lam, nn.MSELoss()(lazy_pred_test, y_train[test]).item()])
    errors = pd.DataFrame(errors, columns=['lam', 'mse'])
    lam = errors.groupby(['lam']).mse.mean().sort_values().index[0]
    print(lam)
    lazy_pred_train, lazy_pred_test = lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info,
                                                   X_train_change, y_train, X_test_change, lam)
    return lazy_pred_train, lazy_pred_test, errors