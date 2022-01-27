import torch
import torch.nn as nn
import time
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import tqdm

from utils import *
from data_generating_funcs import *


def full_model(X, y, hidden_layers=[50], seed=711, tol=1e-3):
    n, p = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = seed)
    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    train_loader = DataLoader(trainset, batch_size=256)

    full_nn = NN4vi(p, hidden_layers, 1)
    early_stopping = EarlyStopping('val_loss', min_delta=tol)
    trainer = pl.Trainer(callbacks=[early_stopping])
    t0 = time.time()
    with io.capture_output() as captured: trainer.fit(full_nn, train_loader, train_loader)
    full_time = time.time() - t0
    test_loss = nn.MSELoss()(full_nn(X_test), y_test).item()
    return full_nn, full_time, test_loss


def dropout_vi(X, y, j, full_nn, seed=711):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    X_test_change = dropout(X_test, j)
    dr_pred_test = full_nn(X_test_change)
    dr_vi = nn.MSELoss()(dr_pred_test, y_test).item() - nn.MSELoss()(full_nn(X_test), y_test).item()
    return dr_vi


def retrain_vi(X, y, j, full_nn, hidden_layers=[50], seed=711, tol=1e-3):
    t0 = time.time()
    _, p = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    X_test_change = dropout(X_test, j)
    X_train_change = dropout(X_train, j)

    retrain_pred_train, retrain_pred_test = retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change,
                                                    tol=tol)
    retrain_time = time.time() - t0
    vi_retrain = nn.MSELoss()(retrain_pred_test, y_test).item() - nn.MSELoss()(y_test, full_nn(X_test)).item()

    # variance
    eps_j = ((y_test - retrain_pred_test) ** 2).detach().numpy().reshape(1, -1)
    eps_full = ((y_test - full_nn(X_test)) ** 2).detach().numpy().reshape(1, -1)
    se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])
    return vi_retrain, retrain_time, se


def lazy_vi(X, y, j, full_nn, hidden_layers=[50], seed=711, lambda_path=np.logspace(0, 2, 10)):
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    X_test_change = dropout(X_test, j)
    X_train_change = dropout(X_train, j)

    t0 = time.time()
    lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(full_nn, X_train_change, X_test_change, y_train,
                                                                hidden_layers, lam_path=lambda_path)
    lazy_time = time.time() - t0
    lazy_train_loss = nn.MSELoss()(lazy_pred_train, y_train).item()
    lazy_test_loss = nn.MSELoss()(lazy_pred_test, y_test).item()
    lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(full_nn(X_test), y_test).item()

    # variance
    eps_j = ((y_test - lazy_pred_test) ** 2).detach().numpy().reshape(1, -1)
    eps_full = ((y_test - full_nn(X_test)) ** 2).detach().numpy().reshape(1, -1)
    se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])

    return lazy_vi, lazy_time, se


def double_lazy_vi(X, y, j, full_nn, hidden_layers=[50], seed=711, lambda_path=np.logspace(0, 2, 10)):
    t0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    X_test_change = dropout(X_test, j)
    X_train_change = dropout(X_train, j)

    t0 = time.time()
    lazy_fullpred_train, lazy_fullpred_test, _ = lazy_train_cv(full_nn, X_train, X_test, y_train,
                                                                hidden_layers, lam_path=lambda_path)
    lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(full_nn, X_train_change, X_test_change, y_train,
                                                                hidden_layers, lam_path=lambda_path)
    lazy_time = time.time() - t0

    lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(lazy_fullpred_test, y_test).item()

    return lazy_vi, lazy_time



