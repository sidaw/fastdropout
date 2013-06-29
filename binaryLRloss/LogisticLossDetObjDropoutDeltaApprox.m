function [nll,g] = LogisticLossDetObjDropoutDeltaApprox(w,X,y,ps)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
pvec = ps * ones(p, 1);
% uncomment next line to never drop out the bias with a different
% probability
% pvec(1) = 0.8;
alpha = 1;

EXw = X*(w .* pvec);
X2 = X.*X;
VarXw = alpha.*(X2) * (w.*w.*pvec.*(1-pvec));
% SigmaXw = sqrt(SigmaXw2);
% randn('seed', 5)
sigyEx = sigmoid(y.*EXw);
nlldet = sum(-log( sigyEx )) + sum(0.5.*sigyEx.*(1-sigyEx).*VarXw);
dmu = -y.*(1-sigyEx);
dsigma = sigyEx.*(1-sigyEx); % *0.5
dEx = pvec .* (dmu'*X)';
dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
g =  dEx + dS2;
nll = nlldet;
% disp(b)
% nll = sum(mean( log1plusexp(-yXw), 2));

