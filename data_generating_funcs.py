import torch
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import Optional
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import os

# Basic data generation
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

# Flexible data modules for more complicated/principled stuff
class FlexDataModule(pl.LightningDataModule):
    def __init__(self, X, y, seed=711, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.33, random_state=self.seed)
        trainset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1,1))
        n = X_train.shape[0]
        n_train = int(np.floor(n*.8))
        n_val = int(n - n_train)
        self.train, self.val = random_split(trainset, [n_train, n_val])

        self.test = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1,1))

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

# MNIST data module with dropout functionality
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dropout=None, chunk_size=0, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.dropout = dropout
        self.chunk_size = chunk_size

    def prepare_data(self):
        # download only
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, targets2keep=[3, 8], stage: Optional[str] = None):
        # transform
        #transform = transforms.Compose([transforms.Resize(14), transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transform)

        if self.dropout is not None:
            if self.chunk_size > 0:
                j, k = self.dropout
                mnist_train.data[:, j:j+self.chunk_size, k:k+self.chunk_size] = 0
                mnist_test.data[:, j:j+self.chunk_size, k:k+self.chunk_size] = 0
            else:
                for (j,k) in self.dropout:
                    mnist_train.data[:, j, k] = 0
                    mnist_test.data[:, j, k] = 0

        # keep the relevant classes
        ix_train = np.array([])
        ix_test = np.array([])
        for t in targets2keep:
            ix_train = np.append(ix_train, np.where(mnist_train.targets == t)[0])
            ix_test = np.append(ix_test, np.where(mnist_test.targets == t)[0])

        #np.random.shuffle(ix_train)
        #np.random.shuffle(ix_test)

        mnist_train.data = mnist_train.data[ix_train]
        mnist_test.data = mnist_test.data[ix_test]

        y_train = mnist_train.targets[ix_train]
        y_test = mnist_test.targets[ix_test]

        mnist_train.targets = torch.tensor([1 if y == targets2keep[1] else 0 for y in y_train],
                                           dtype=torch.float32).view(-1, 1)
        mnist_test.targets = torch.tensor([1 if y == targets2keep[1] else 0 for y in y_test],
                                          dtype=torch.float32).view(-1, 1)

        # train/val split
        n = mnist_train.data.shape[0]
        n_train = int(np.floor(n * .8))
        n_val = int(n - n_train)
        mnist_train, mnist_val = random_split(mnist_train, [n_train, n_val])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
