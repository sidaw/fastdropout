function [nll,g] = LogisticLossDetObjDropout(w,X,y,ps)
% This is eq 8) of the ICML paper
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
pvec = ps * ones(p, 1);
% uncomment next line to never drop out the bias
% pvec(1) = 0.8;
alpha = 1;

EXw = X*(w .* pvec);
X2 = X.*X;
SigmaXw2 = alpha.*(X2) * (w.*w.*pvec.*(1-pvec));
% SigmaXw = sqrt(SigmaXw2);
% randn('seed', 5)
kappa = 0.125*pi;
one_plus_sigma = (1 + kappa.*SigmaXw2);
c = 1./sqrt(one_plus_sigma);

if 0
sig_cyEXw = 1./(1+exp( c.*y.*EXw ));
alpha = bsxfun(@times, sig_cyEXw, y);
beta = bsxfun(@times, -y.*c.*sig_cyEXw.*(1-sig_cyEXw), y);

numsample = 6000;
sigmar =  bsxfun(@times, SigmaXw, randn(n, numsample));
Xw = bsxfun(@plus, EXw, sigmar);
yXw =  bsxfun(@times, Xw, y);
expyXw = exp(yXw);
alpha1 = mean(bsxfun(@times, 1./(1+expyXw), y), 2);
beta1 = mean(bsxfun(@times, sigmar./(1+expyXw), y), 2) ./ SigmaXw2;
changewp = (w.*(1-pvec))';
meandiff = bsxfun(@times, X,  changewp);
meandiff = bsxfun(@times, meandiff', beta');
end


% nllgauss = sum(mean(log(1+1./expyXw), 2));
sigycEx = sigmoid(y.*c.*EXw);
nlldet = -sum(log( sigycEx ) ./ c);
dmu = -y.*(1-sigycEx);

dsigma = alpha.*kappa.*c.*( y.*EXw.*(1-sigycEx).*c...
    - log(sigycEx)); % *0.5
dEx = pvec .* (dmu'*X)';
dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
g =  dEx + dS2;
    

% nlldetnaive = -sum(log( sigmoid(y.*c.*EXw) ));
% b = [nllgauss nlldet nlldetnaive ];
nll = nlldet;
% disp(b)
% nll = sum(mean( log1plusexp(-yXw), 2));

