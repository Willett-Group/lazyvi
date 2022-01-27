import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings('ignore')

import sys, os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import *
from classification_utils import *
from data_generating_funcs import *


parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--n', type=int, default=1000)
args = parser.parse_args()


true_vi = {
    'X1': .136,
    'X2': .236,
    'X3': 0,
    'X4': 0
}

results = []
niter = args.niter

for n in [2500, 5000]:
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
        acc0 = torch.sum(y_test == y_hat).item() / (len(y_test) * 1.0)
        results.append(['full_model', 'full', acc0, 0])
        yhat_full = full_nn(X_test).detach().numpy().round()
        y_test_np = y_test.detach().numpy()

        n, p = X.shape
        for j in range(p):
            # reduced dataset
            Xj = dropout(X, j)
            Xj_train = dropout(X_train, j)
            Xj_test = dropout(X_test, j)

            # lazy training
            t0  = time.time()
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
            lazy_time = time.time() - t0
            yhat_lazy = lazy_pred_test.detach().numpy().round()

            accj = accuracy_score(y_test_np, yhat_lazy)
            se = get_accuracy_se(y_test_np, yhat_full, yhat_lazy)
            results.append([f'X{j+1}', 'lazy', acc0-accj, se, lazy_time])

    df = pd.DataFrame(results, columns=['varr', 'method', 'vi', 'se', 'time'])
    df['true_vi'] = df.varr.map(true_vi)
    df.to_csv(f'results/williamson4_n{n}_iter{niter}.csv', index=False)

