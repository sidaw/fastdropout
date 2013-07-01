function [tnll,g] = LogisticOnevsAllLossDetObjDropout(W,X,Y,ps)
% W(feature,class)
% X(instance,feature)
% Y(instance,class) in 1 of K encoding
K = size(Y,2);
W = reshape(W, [], K);
tnll = 0;
g = zeros(size(W));
for k = 1:K
    w = W(:,k);
    [n,p] = size(X);
    y = 2*Y(:,k)-1;
    pvec = ps * ones(p, 1);
    % uncomment next line to dropout bias at a different rate
    % pvec(1) = 0.8;
    alpha = 1;
    EXw = X*(w .* pvec);
    X2 = X.*X;
    VarXw = alpha.*(X2) * (w.*w.*pvec.*(1-pvec));
    % SigmaXw = sqrt(SigmaXw2);
    % randn('seed', 5)
    sigyEx = sigmoid(y.*EXw);
    nlldet = sum(-log( sigyEx )) + sum(0.5.*sigyEx.*(1-sigyEx).*VarXw);
    dmu = -y.*(1-sigyEx) + y.*0.5.*(sigyEx.*(1-sigyEx).*(1-2*sigyEx)).*VarXw;
    dsigma = sigyEx.*(1-sigyEx); % *0.5
    dEx = pvec .* (dmu'*X)';
    dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
    g(:,k) =  dEx + dS2;
    tnll = tnll + nlldet;
end
g=g(:);