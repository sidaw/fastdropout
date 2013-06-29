function [tnll,g] = LogisticOnevsAllLoss(W,X,Y)
% w(feature,1)
% X(instance,feature)
% y(instance,1)
K = size(Y,2);
W = reshape(W, [], K);
tnll = 0;
g = zeros(size(W));
ps = 1;

for k = 1:K
    w = W(:,k);
    [n,p] = size(X);
    y = 2*Y(:,k)-1;
    pvec = ps * ones(p, 1);
    EXw = X*(w .* pvec);

    sigyEx = sigmoid(y.*EXw);
    nlldet = sum(-log( sigyEx ));
    dmu = -y.*(1-sigyEx);
    dEx = pvec .* (dmu'*X)';
    g(:,k) =  dEx;
    tnll = tnll + nlldet;
end
g=g(:);