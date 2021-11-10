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

lam_df = pd.DataFrame()
for lam in np.linspace(1, 30, 10):
    print(lam)
    X, y = generate_linear_data(beta, corr=rho)
    for t in range(100):
        df = vi_experiment_wrapper(X, y, 50, [0], lam=lam, do_retrain=False)
        df['lam'] = lam
        df['true_vi'] = beta[0]**2*(1-rho**2)
        lam_df = lam_df.append(df)

lam_df.to_csv(f'results/lambda_coverage_width{50}_iter100.csv', index=False)