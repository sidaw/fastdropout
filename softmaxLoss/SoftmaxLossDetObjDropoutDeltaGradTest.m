function [tnll,g] = SoftmaxLossDetObjDropoutDeltaGradTest(W,X,Y,ps)
% W(feature,class)
% X(instance,feature)
% Y(instance,class) in 1 of K encoding
K = size(Y,2);
[n,p] = size(X);
pvec = ps * ones(p, 1);
% uncomment next line to dropout bias at a different rate
% pvec(1) = 0.8;
    

W = reshape(W, [], K);
tnll = 0;
g = zeros(size(W));

Xw = X*bsxfun(@times, pvec, W);
% Xw = X*W;

normalize = bsxfun(@minus, Xw, max(Xw,[],2) );
expXw = exp(normalize);
Z = sum(expXw,2);
expXwdZ = bsxfun(@rdivide, expXw, Z);

for k = 1:K
    w = W(:,k);
    y = Y(:,k);
    Pk = expXwdZ(:,k);
    
    X2 = X.*X;
    VarXw = (X2) * (w.*w.*pvec.*(1-pvec));
    % SigmaXw = sqrt(SigmaXw2);
    % randn('seed', 5)
    
    nlldet = k*sum(Pk);
    dmu =  bsxfun(@times, -Pk,expXwdZ);
    dmu(:,k) = dmu(:,k) + Pk;
    dmu = k*dmu;
    dsigma = 0*Pk.*(1-Pk); % *0.5
    % small efficiency trick to not transpose large matrix
    dEx = bsxfun(@times,pvec,(dmu'*X)');
    dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
    g = g + dEx;
    tnll = tnll + nlldet;
end
g=g(:);