import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore')

import sys, os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import *
from data_generating_funcs import *


beta = [1.5, 1.2, 1, 0, 0, 0]
rho = .75
m = 50

true_vi = {'X1':beta[0]**2*(1-rho**2),
          'X2':beta[1]**2*(1-rho**2),
          'X3':1,
          'X4':0,
          'X5':0,
          'X6':0}

cvg = pd.DataFrame()


for i in range(10):
    W, V = generate_WV(beta, 12)
    X, y = generate_2lnn_data(W, V, 5000, corr=rho)
    df = vi_experiment_wrapper(X, y, m, exp_iter=i, lambda_path=np.logspace(1, 3, 15))
    #df['true_vi'] = df.variable.map(true_vi)
    cvg = cvg.append(df)

cvg.to_csv(f'results/coverage_width{50}_iter10_2lnn_noisy.csv', index=False)