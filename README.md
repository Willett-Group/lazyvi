variable importance estimation using lazy training
==============================

`vi_experiments.py` produces simulation results for various data generating functions and network architectures using
the dropout, retraining, and lazy training methods. 

Optional parameters:
`--corr` correlation between variables $X_1$ and $X_2$
`--width` width of hidden layer in 2-layer training network
`--data` data generating function
`--tol` network convergence criterion
`--niter` number of repetitions of experiment

For example, to run 10 iterations of an experiment with a 100-width network on data generated from a linear model 
with $corr(X_1, X_2) = .75$, run:

`python vi_experiments.py --width 100 --corr .75 --niter 10 --data linear`
