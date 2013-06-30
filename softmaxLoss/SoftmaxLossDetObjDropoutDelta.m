function [tnll,g] = SoftmaxLossDetObjDropoutDelta(W,X,Y,ps)
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
X2 = X.*X;

normalize = bsxfun(@minus, Xw, max(Xw,[],2) );
expXw = exp(normalize);
Z = sum(expXw,2);
expXwdZ = bsxfun(@rdivide, expXw, Z);

for k = 1:K
    w = W(:,k);
    y = Y(:,k);
    Pk = expXwdZ(:,k);
    
    VarXw = (X2) * (w.*w.*pvec.*(1-pvec));
    % SigmaXw = sqrt(SigmaXw2);
    % randn('seed', 5)
    
    nlldet = sum(-y.*log( Pk )) + sum(0.5.*Pk.*(1-Pk).*VarXw);
    dmu = -(y-Pk);
    
    dPk =  bsxfun(@times, -Pk,expXwdZ);
    dPk(:,k) = dPk(:,k) +  Pk;
     dPk1minusPk = bsxfun(@times, dPk, 0.5.*(1-Pk).*VarXw)+...
         bsxfun(@times, -dPk, 0.5.*Pk.*VarXw);
    
    %dPk1minusPk = bsxfun(@times, dPk, 0.5.*VarXw); 
    dEx = pvec .* (dmu'*X)';
    
    dsigma = Pk.*(1-Pk); % *0.5
    dS2 = w.*pvec.*(1-pvec) .* (dsigma'*X2)'; % *2
    g(:,k) =  dEx + dS2;
    g = g + bsxfun(@times,pvec,(dPk1minusPk'*X)');
    
    tnll = tnll + nlldet;
end
g=g(:);