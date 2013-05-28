function [nll,g] = LogisticLossMCDropoutSampleGaussNumDiff(w,X,y,ps, numsample)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
pvec = ps * ones(p, 1);
% uncomment next line to never drop out the bias
pvec(1) = 0.8;
EXw = X*(w .* pvec);
SigmaXw = sqrt( (X.*X) * (w.*w.*pvec.*(1-pvec)) );
% yEXw = y .* EXw;
randn('seed', 5)
sigmarnad =  bsxfun(@times, SigmaXw, randn(n, numsample));
sig_inputs = bsxfun(@plus, EXw, sigmarnad);
eps = 0.05;
sig_inputs2 = bsxfun(@plus, EXw+eps, sigmarnad);

yXw =  bsxfun(@times, sig_inputs, y);
expyXw = exp(yXw);
Ecase = mean(bsxfun(@times, 1./(1+expyXw), y), 2);

yXw2 =  bsxfun(@times, sig_inputs2, y);
expyXw2 = exp(yXw2);
Ecase2 = mean(bsxfun(@times, 1./(1+expyXw2), y), 2);
numdiff = (Ecase2 - Ecase) / eps;

meandiff = bsxfun(@times, X, (w.*(1-pvec))' );
meandiff = bsxfun(@times, meandiff', numdiff');

g = -pvec .* ( X.'*Ecase+sum(X.' .* meandiff,2) );

nll = sum(mean(log(1+1./expyXw), 2));
% nll = sum(mean( log1plusexp(-yXw), 2));
