fastdropout
===========
Loss functions for fast dropout/dropout and baselines like softmax, and logistic regressions
They are described in the readme file in the respective folders.
To get the idea behind them, see the fast dropout paper on my website at stanford.edu/~sidaw

If you have minFunc, you can try running run_experiment.m
The example_data loaded are vectors from the atheism vs. christian newsgroup task.
With the fixed random seed, I got:
DetDropout: 0.882188
Dropout: 0.882188
LR: 0.827489

Dependency on minFunc:

minFunc can be found here:
http://www.di.ens.fr/~mschmidt/Software/minFunc.html

