import torch
import numpy as np
import pandas as pd
import random


def generate_linear_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    random.seed(seed)
    cov = [[1, corr], [corr, 1]]
    beta = np.array(beta, dtype=float)
    p = beta.shape[0]
    VI_true = beta ** 2
    VI_true[0:2] = VI_true[0:2] * (1 - corr ** 2)
    X = np.random.normal(0, 1, size=(N, p))
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)
    normal_noise = np.random.normal(0, sigma, size=N)
    EY = np.matmul(X, beta)
    Y = EY + normal_noise

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    return X, Y

def generate_logistic_data(beta, sigma=0.1, N=5000, seed=1, corr=0.5):
    random.seed(seed)
    cov = [[1, corr], [corr, 1]]
    beta = np.array(beta, dtype=float)
    p = beta.shape[0]
    X = np.random.normal(0, 1, size=(N, p))
    X[:, np.array([0, 1])] = np.random.multivariate_normal([0, 0], cov, size=N)
    normal_noise = np.random.normal(0, sigma, size=N)
    EY = np.exp(X@beta)/(1 + np.exp(X@beta))
    Y = EY + normal_noise

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    return X, Y


def generate_WV(beta, m, V='random', sigma=0.1):
    p = len(beta)
    W = np.zeros((m, p))
    for j, b in enumerate(beta):
        W[:, j] = np.random.normal(b, sigma, size=m)
        # W[:, j] = b
    W = torch.tensor(W, dtype=torch.float32)
    if V=='random':
        V = torch.tensor(np.random.normal(size=(1, m)), dtype=torch.float32)
    elif V == 'sequential':
        V = torch.tensor((np.arange(m)+1)/m, dtype=torch.float32)
    return W, V

def generate_2lnn_data(W, V, n, corr=0.5):
    p = W.shape[1]
    sigma = np.eye(p)
    sigma[0, 1] = corr
    sigma[1, 0] = corr
    X = np.random.multivariate_normal(np.zeros(p), sigma, size=n)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(torch.matmul(V, torch.relu(torch.matmul(W, X.T))).detach().numpy(),
                     dtype=torch.float32)
    return X, Y.reshape(-1, 1)


def generate_non_additive_6(n=5000, rho=.5):
    p = 6
    sigma = np.eye(p)
    sigma[0, 1] = rho
    sigma[1, 0] = rho
    X = np.random.multivariate_normal(np.zeros(p), sigma, size=n)
    Y = X[:,0]*np.sin(X[:,1]+2*X[:,2])*np.cos(X[:,3]+2*X[:,4])
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).reshape(-1,1)
    return X, Y

def generate_floodgate_sims(n=1100,p=1000, rho=.3, s=30, amplitude=5):
    #Sigma = pd.read_csv('../../floodgate/vignettes/Sigma.csv').values
    # make AR(1) covariance
    S = np.zeros((p, p))
    for i in range(p):
        S[i, i:] = rho ** np.arange(p - i)
        for j in range(i):
            S[i, j] = S[j, i]

    ## choose non-null varaibles randomly
    S_star = np.random.choice(np.arange(p), s, replace=False)
    beta = np.zeros(p)
    beta[S_star] = np.random.choice([-1, 1], s) * amplitude / np.sqrt(n)

    ## generate the covaraites X
    X = np.random.multivariate_normal(np.zeros(p), S, size=n)
    ## Generate the response Y from a linear model

    Y = X @ beta + np.random.normal(size=n)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
    return X, Y, beta

def generate_williamson_sims(n,p=4):
    X = np.random.normal(size=(n,p))
    beta = [2.5, 3.5] + [0]*(p-2)
    y = (X@beta + np.random.normal(size=n) > 0).astype(int)
    return X, y