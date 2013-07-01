function [nll,g] = LogisticLossDetObjDropoutDeltaMoreData(w,X,y,ps,Xu,a)
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
VarXw = alpha.*(X2) * (w.*w.*pvec.*(1-pvec));
sigyEx = sigmoid(y.*EXw);
sigEx = sigmoid(EXw);

EXuw = Xu*(w .* pvec);
Xu2 = Xu.*Xu;
VarXuw = alpha.*(Xu2) * (w.*w.*pvec.*(1-pvec));
sigEXu = sigmoid(EXuw);


nu = size(Xu, 1);

b = 1;
s = n/(n+nu*a);
t= n/(n+nu*a);
nlldet = sum(-log( sigyEx )) + b*0.5.*s.*sum(sigEx.*(1-sigEx).*VarXw)...
    + a*0.5.*t.*sum(sigEXu.*(1-sigEXu).*VarXuw);

dmu = -y.*(1-sigyEx) + b*0.5.*s.*(sigEx.*(1-sigEx).*(1-2*sigEx)).*VarXw;
dmuu = a*0.5.*t.*(sigEXu.*(1-sigEXu).*(1-2*sigEXu)).*VarXuw;
dsigma = b*s.*sigyEx.*(1-sigyEx); % *0.5
dsigmau = a*t.*sigEXu.*(1-sigEXu); % *0.5
dEx = pvec .* ((dmu'*X)' + (dmuu'*Xu)');
dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
dS2u = w.*pvec.*(1-pvec) .* (dsigmau'*Xu2)';
g =  dEx + dS2 + dS2u;
nll = nlldet;
% disp(b)
% nll = sum(mean( log1plusexp(-yXw), 2));

