import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore')

import sys, os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils import vi_experiment_wrapper
from data_generating_funcs import generate_linear_data


beta = [1.5, 1.2, 1, 0, 0, 0]
rho = .75
m = 50

true_vi = {'X1':beta[0]**2*(1-rho**2),
          'X2':beta[1]**2*(1-rho**2),
          'X3':1,
          'X4':0,
          'X5':0,
          'X6':0}

beta = [1.5, 1.2, 1, 0, 0, 0]
j = 3

corr_df = pd.DataFrame()
for rho in np.linspace(0, 1, 10):
    print(rho)
    for t in range(10):
        X, y = generate_linear_data(beta, corr=rho)
        df = vi_experiment_wrapper(X, y, 50, exp_iter=t, ix=[j-1])
        df['corr'] = rho
        df['true_vi'] = beta[j-1] ** 2 * (1 - rho ** 2)
        corr_df = corr_df.append(df)

corr_df.to_csv(f"results/correlation_exp_x{j}.csv", index=False)