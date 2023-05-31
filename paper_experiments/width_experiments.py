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

cvg = pd.DataFrame()

for i in range(50):
    if i % 10 == 0:
        cvg.to_csv(f'results/tmp_width_exp.csv', index=False)
    X, y = generate_linear_data(beta, corr=rho)
    for m in [1, 5, 10, 20, 30, 40, 50, 75, 100]:
        df = vi_experiment_wrapper(X, y, m, exp_iter=i, ix=[0,1,2], lambda_path=np.logspace(1, 3, 15))
        df['true_vi'] = df.variable.map(true_vi)
        df['width'] = m
        cvg = cvg.append(df)

cvg.to_csv(f'results/width_experiment_iter100_15cv.csv', index=False)