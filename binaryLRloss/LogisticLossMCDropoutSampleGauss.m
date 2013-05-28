function [nll,g] = LogisticLossMCDropoutSampleGauss(w,X,y,ps, numsample)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
pvec = ps * ones(p, 1);
% uncomment next line to never drop out the bias
% pvec(1) = 1;
EXw = X*(w .* pvec);
SigmaXw = sqrt( (X.*X) * (w.*w.*pvec.*(1-pvec)) );
% yEXw = y .* EXw;

sigmarnad =  bsxfun(@times, SigmaXw, randn(n, numsample));
sig_inputs = bsxfun(@plus, EXw, sigmarnad);
yXw =  bsxfun(@times, sig_inputs, y);
expyXw = exp(yXw);
Ecase = mean(bsxfun(@times, 1./(1+expyXw), y), 2);
g = -pvec .* (X.'* Ecase);

nll = sum(mean( log1plusexp(-yXw), 2));
