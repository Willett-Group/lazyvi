import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings('ignore')

import sys, os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from old_scripts.classification_utils import *
from data_generating_funcs import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=10)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--rho', type=float, default=0)
args = parser.parse_args()


true_vi = {
    'X1': .136,
    'X2': .236,
    'X3': 0,
    'X4': 0
}

results = []
niter = args.niter
n = args.n
rho = args.rho

def williamson4_corr(n,p=4, rho=0):
    Sigma = np.eye(p)
    Sigma[0,1] = rho
    Sigma[1,0] = rho
    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
    beta = [2.5, 3.5] + [0]*(p-2)
    y = (X@beta + np.random.normal(size=n) > 0).astype(int)
    #y = bernoulli()
    return X, y

for t in range(niter):
    X, y = williamson4_corr(n, p=4, rho=rho)
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

    # numpy versions for other methods
    X_train_np = X_train.detach().numpy()
    y_train_np = y_train.detach().numpy()
    X_test_np = X_test.detach().numpy()
    y_test_np = y_test.detach().numpy()

    # fit full models for lg/rf
    lg = LogisticRegression()
    rf = RandomForestClassifier()

    lg.fit(X_train_np, y_train_np)
    rf.fit(X_train_np, y_train_np)

    lg_full_pred = lg.predict(X_test_np).round()
    acc_lg = accuracy_score(y_test_np, lg_full_pred)

    rf_full_pred = rf.predict(X_test_np).round()
    acc_rf = accuracy_score(y_test_np, rf_full_pred)

    n, p = X.shape
    for j in range(p):
        # reduced dataset
        Xj = dropout(X, j)
        Xj_train = dropout(X_train, j)
        Xj_test = dropout(X_test, j)

        Xj_train_np = dropout(X_train_np, j)
        Xj_test_np = dropout(X_test_np, j)

        # Logistic
        t0 = time.time()
        lgj = LogisticRegression()
        lgj.fit(Xj_train_np, y_train)
        acc_lgj = accuracy_score(y_test_np, lgj.predict(Xj_test_np).round())
        results.append([f'X{j + 1}', 'logistic', acc_lg - acc_lgj, 0, time.time() - t0])

        # RF
        t0 = time.time()
        rfj = RandomForestClassifier()
        rfj.fit(Xj_train_np, y_train)
        acc_rfj = accuracy_score(y_test_np, rfj.predict(Xj_test_np).round())
        results.append([f'X{j + 1}', 'rf', acc_rf - acc_rfj, 0, time.time() - t0])

        # dropout
        yhat_dr = full_nn(Xj_test).detach().numpy().round()
        accj = accuracy_score(y_test_np, yhat_dr)
        se = get_accuracy_se(y_test_np, yhat_full, yhat_dr)
        results.append([f'X{j + 1}', 'dropout', acc0 - accj, se, 0])


        # retrain
        retrain_accuracy, retrain_time, y_hat_ret = retrain_vi_bin(Xj, y, j)
        y_hat_ret_np = y_hat_ret.detach().numpy().round()

        accj = accuracy_score(y_test_np, y_hat_ret_np)
        se = get_accuracy_se(y_test_np, yhat_full, y_hat_ret_np)
        results.append([f'X{j+1}', 'retrain', acc0-accj, se, retrain_time])

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
df.to_csv(f'results/williamson4_n{n}_iter{niter}_rho{rho}_comparison.csv', index=False)

