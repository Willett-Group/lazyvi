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

parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, default=12)
parser.add_argument('--niter', type=int, default=1)
parser.add_argument('--corr', type=float, default=0.5)
parser.add_argument('--data', type=str, default='linear')
args = parser.parse_args()


beta = [1.5, 1.2, 1, 0, 0, 0]
sigma = 0.1
N = 5000
seed = 1
p = len(beta)
hidden_layers = [args.width]
corr = args.corr
tol = 1e-2
ix = [0,1,2,3]

exp_name = '{}_width{}_corr{}_ix{}_iter{}'.format(args.data, args.width, args.corr, '|'.join([str(x) for x in ix]), args.niter)

results = []
cv_results = pd.DataFrame()

if args.data == '2lnn':
    # generate weights up front for all sims
    W, V = generate_WV(beta, 12)

for t in tqdm.tqdm(np.arange(args.niter)):
    if args.data == 'linear':
        X, y = generate_linear_data(beta, N=N, seed=t, corr=corr)
    elif args.data == 'logistic':
        X, y = generate_logistic_data(beta, N=N, seed=t, corr=corr)
    elif args.data == '2lnn':
        X, y = generate_2lnn_data(W, V, N, corr=corr)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1)
    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    train_loader = DataLoader(trainset, batch_size=256)

    full_nn = NN4vi(p, hidden_layers, 1)
    #tb_logger = pl.loggers.TensorBoardLogger('logs/{}'.format(exp_name), name='full')#, default_hp_metric=False)
    early_stopping = EarlyStopping('val_loss', min_delta=tol)
    trainer = pl.Trainer(callbacks=[early_stopping])
    t0 = time.time()
    trainer.fit(full_nn, train_loader, train_loader)
    full_time = time.time() - t0
    full_pred_test = full_nn(X_test)
    results.append(['all', 'full model', full_time, 0,
                    nn.MSELoss()(full_nn(X_train), y_train).item(),
                    nn.MSELoss()(full_pred_test, y_test).item()])

    for j in ix:
        varr = 'X' + str(j + 1)
        # DROPOUT
        X_test_change = dropout(X_test, j)
        X_train_change = dropout(X_train, j)
        dr_pred_train = full_nn(X_train_change)
        dr_pred_test = full_nn(X_test_change)
        dr_train_loss = nn.MSELoss()(dr_pred_train, y_train).item()
        dr_test_loss = nn.MSELoss()(dr_pred_test, y_test).item()
        dr_vi = nn.MSELoss()(dr_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()
        results.append([varr, 'dropout', 0, dr_vi, dr_train_loss, dr_test_loss])

        # LAZY
        t0 = time.time()
        lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(full_nn, X_train_change, X_test_change, y_train,
                                                             hidden_layers, lam_path = np.logspace(0, 2, 10))
        lazy_time = time.time() - t0
        lazy_train_loss = nn.MSELoss()(lazy_pred_train, y_train).item()
        lazy_test_loss = nn.MSELoss()(lazy_pred_test, y_test).item()
        lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()
        results.append([varr, 'lazy', lazy_time, lazy_vi, lazy_train_loss, lazy_test_loss])
        cv_res['variable'] = varr
        cv_res['iter'] = t
        cv_results = cv_results.append(cv_res, ignore_index=True)

        # RETRAIN
        t0 = time.time()
        retrain_pred_train, retrain_pred_test = retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=tol)
        retrain_time = time.time() - t0
        vi_retrain = nn.MSELoss()(retrain_pred_test, y_test).item() - nn.MSELoss()(y_test, full_pred_test).item()
        loss_rt_test = nn.MSELoss()(retrain_pred_test, y_test).item()
        loss_rt_train = nn.MSELoss()(retrain_pred_train, y_train).item()
        results.append([varr, 'retrain', retrain_time, vi_retrain, loss_rt_train, loss_rt_test])

df = pd.DataFrame(results, columns=['variable', 'method', 'time', 'vi', 'train_loss', 'test_loss'])
cv_results.to_csv('results/cv/{}.csv'.format(exp_name), index=False)
df.to_csv('results/{}.csv'.format(exp_name), index=False)
