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

kappa = 0.125*pi;
one_plus_sigma = (1 + kappa.*SigmaXw2);
c = 1./sqrt(one_plus_sigma);

sigycEx = sigmoid(y.*c.*EXw);
nlldet = -sum(log( sigycEx ) ./ c);
dmu = -y.*(1-sigycEx);

dsigma = alpha.*kappa.*c.*( y.*EXw.*(1-sigycEx).*c...
    - log(sigycEx)); % *0.5
dEx = pvec .* (dmu'*X)';
dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
g =  dEx + dS2;
nll = nlldet;

