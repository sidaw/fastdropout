# fastdropout

Loss functions for fast dropout/dropout and baselines like softmax, and logistic regressions
They are described in the readme file in the respective folders.
To get the idea behind them, see the fast dropout paper on my website at nlp.stanford.edu/~sidaw

If you have minFunc, you can try running run_experiment.m
The example_data loaded are vectors from the atheism vs. christian newsgroup task.
With the fixed random seed, I got:

```
DetDropout: 0.882188
Dropout: 0.882188
LR: 0.827489
```

## Loss functions

### Loss for logistic regression
These loss functions take weight w, data X, labels y and return the negative loglikihood nll and the gradient g.

These are all binary loss functions and they can be found in ./binaryLRloss
Similar multiclass loss functions are in ./softmaxLoss

```matlab

% w: model weights
% X: data matrix, dimdata by numdata
% y: binary label vector
% ps: dropout rate

function [nll,g,H,T] = LogisticLoss(w,X,y)
% Baseline loss function from PMTK, with some optimizations

function [nll,g,H] = LogisticLossMCDropout(w,X,y,ps)
% Monte Carlo real dropout, single sample

function [nll,g,H] = LogisticLossMCDropoutSample(w,X,y,ps, miniter, numiter)
% Monte Carlo real dropout, numiter samples

function [nll,g] = LogisticLossDetObjDropout(w,X,y,ps)
% This is eq 8) of the ICML paper, intergrating the Gaussian

function [nll,g] = LogisticLossDetObjDropoutDelta(w,X,y,ps)
% Dropout loss via the Delta approximation, somewhat less accurate

function [nll,g] = LogisticLossMCDropoutSampleGauss(w,X,y,ps, numsample)
% Sample from the Gaussian

function [nll,g] = LogisticLossMCDropoutSampleGaussIntegrate(w,X,y,ps, numsample)
% Differenitate inside the expectation and then intergrate

function [nll,g] = LogisticLossMCDropoutSampleGaussNumDiff(w,X,y,ps, numsample)
% Use numerical diff to compute derivative
```

### Loss for 2 layer neural networks
```matlab

% W: weights
% X: data matrix, dimdata by numdata
% k: number of classes
% decode: information on how to decode w
% params: things like
%    params.numsamples = 50; 
%    params.p = 0.5; % dropout rate
%    params.p1 = 1; %dropout rate for bias
% y1ofn: labels in 1 out of n encoding, so numclass by numdata

function [nll,g, expHwdZ, labels] = Softmax2NN(W,X,k,decode,params,y1ofn)
% Baseline softmax classification with 2 layer NN

function [nll,g, expHwdZ, labels] = Softmax2NN_MCDropoutMulti(W,X,k,decode,params,y1ofn)
% Real dropout 2NN take many samples

function [se, g, yhat] = Linear2NNDet(W,X,k,decode,params,y)
% 2 layer NN for regression with fast dropout

function [nll,g, Mu3, labels] = LogisticOneVsAllDetLoss(W,X,k,decode, params,y1ofn)
% 2 layer NN for 1 vs. all classification with dropout

function [nll,g, EexpdZ, labels] = Softmax2NNLossFastSample(W,X,k,decode,params,y1ofn)
% Sample from the Gaussian

function [nll,g, weightedsum, labels] = Softmax2NNLossFastSigmaPoints(W,X,k,decode,params,y1ofn)
% Use Sigma point
```

minFunc can be found here:
http://www.di.ens.fr/~mschmidt/Software/minFunc.html

License: [MIT](LICENSE)
