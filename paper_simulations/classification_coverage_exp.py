import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


import warnings

warnings.filterwarnings('ignore')

import sys, os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import *
from classification_utils import *
from data_generating_funcs import *

true_vi = {
    'X1': .136,
    'X2': .236,
    'X3': 0,
    'X4': 0
}

results = []
niter = 10
n = 1000

for t in range(niter):
    X, y = williamson4(n, p=4)
    data_module = FlexDataModule(X, y)
    data_module.setup()

    # extract train/test from data_module
    X_train = data_module.train.dataset.tensors[0]
    y_train = data_module.train.dataset.tensors[1]

    X_test = data_module.test.tensors[0]
    y_test = data_module.test.tensors[1]

    early_stopping = EarlyStopping('my_loss_epoch', min_delta=1e-3)

    trainer = pl.Trainer(callbacks=early_stopping, max_epochs=100)
    full_nn = BinClass()
    trainer.fit(full_nn, data_module)

    probs = full_nn(data_module.test.tensors[0])
    y_hat = torch.round(probs)
    y_test = data_module.test.tensors[1]
    acc0 = torch.sum(y_test == y_hat).item() / (len(y_test) * 1.0)
    results.append(['full_model', acc0, 0])

    n, p = X.shape
    for j in range(p):
        # reduced dataset
        Xj_train = dropout(X_train, j)
        Xj_test = dropout(X_test, j)

        # lazy training
        kf = KFold(n_splits=5, shuffle=True)
        errors = []
        grads, flat_params, shape_info = extract_grad(Xj_train, full_nn)
        # print(grads.shape)

        for lam in np.logspace(-1,2,20):
            for train, test in kf.split(Xj_train):
                dr_pred_train = full_nn(Xj_train[train])
                grads_train = grads[train]
                lazy_pred_train, lazy_pred_test = lazy_predict_bin(grads_train, flat_params, full_nn, shape_info,
                                                                   Xj_train[train], y_train[train],
                                                                   Xj_train[test], lam)
                errors.append([lam, nn.MSELoss()(lazy_pred_test, y_train[test]).item()])
        errors = pd.DataFrame(errors, columns=['lam', 'mse'])
        lam = errors.groupby(['lam']).mse.mean().sort_values().index[0]
        #print(lam)
        lazy_pred_train, lazy_pred_test = lazy_predict_bin(grads, flat_params, full_nn, shape_info,
                                                           Xj_train, y_train, Xj_test, lam)

        # compute std errors for ci
        # DO CROSS ENTROPY HERE
        eps_j = ((y_test - lazy_pred_test) ** 2).detach().numpy().reshape(1, -1)
        eps_full = ((y_test - full_nn(X_test)) ** 2).detach().numpy().reshape(1, -1)
        se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])

        y_hat = lazy_pred_test.round()
        accj = torch.sum(y_test == y_hat).item() / (len(y_test) * 1.0)

        results.append([f'X{j+1}', acc0-accj, se])

df = pd.DataFrame(results, columns=['varr', 'vi', 'se'])
df['true_vi'] = df.varr.map(true_vi)
df.to_csv(f'results/williamson4_n{n}_iter{niter}.csv', index=False)

