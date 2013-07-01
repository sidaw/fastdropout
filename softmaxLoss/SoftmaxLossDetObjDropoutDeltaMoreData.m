function [tnll,g] = SoftmaxLossDetObjDropoutDeltaMoreData(W,X,Y,ps,Xu,a)
% W(feature,class)
% X(instance,feature)
% Xu(instance, feature) unlabeled data
% Y(instance,class) in 1 of K encoding
K = size(Y,2);
[n,p] = size(X);
nu = size(Xu, 1);
pvec = ps * ones(p, 1);
% uncomment next line to dropout bias at a different rate
% pvec(1) = 0.8;

W = reshape(W, [], K);
tnll = 0;
g = zeros(size(W));

Xw = X*bsxfun(@times, pvec, W);
X2 = X.*X;
Xw = bsxfun(@minus, Xw, max(Xw,[],2) );
expXw = exp(Xw);
Z = sum(expXw,2);
expXwdZ = bsxfun(@rdivide, expXw, Z);

Xuw = Xu*bsxfun(@times, pvec, W);
Xu2 = Xu.*Xu;
Xuw = bsxfun(@minus, Xuw, max(Xuw,[],2) );
expXuw = exp(Xuw);
Zu = sum(expXuw,2);
expXuwdZu = bsxfun(@rdivide, expXuw, Zu);

s = n/(n+nu*a);

for k = 1:K
    w = W(:,k);
    y = Y(:,k);
    
    Pk = expXwdZ(:,k);
    VarXw = (X2) * (w.*w.*pvec.*(1-pvec));
    
    Puk = expXuwdZu(:,k);
    VarXuw = (Xu2) * (w.*w.*pvec.*(1-pvec));
    % SigmaXw = sqrt(SigmaXw2);
    % randn('seed', 5)
    
    nlldet = sum(-y.*log( Pk )) + s*sum(0.5.*Pk.*(1-Pk).*VarXw) + ...
        s*a*sum(0.5.*Puk.*(1-Puk).*VarXuw);
    
    dmu = -(y-Pk);
    
    dPk =  bsxfun(@times, -Pk,expXwdZ);
    dPk(:,k) = dPk(:,k) +  Pk;
    dPk1minusPk = bsxfun(@times, dPk, 0.5.*(1-2*Pk).*VarXw);
    
    dPuk =  bsxfun(@times, -Puk,expXuwdZu);
    dPuk(:,k) = dPuk(:,k) +  Puk;
    dPuk1minusPuk = bsxfun(@times, dPuk, 0.5.*(1-2*Puk).*VarXuw);
    
    dEx = pvec .* (dmu'*X)';
    
    dsigma = Pk.*(1-Pk); % *0.5
    dsigmau = Puk.*(1-Puk); % *0.5
    
    dS2 = w.*pvec.*(1-pvec) .* ...
        (s*(dsigma'*X2)' + a*s*(dsigmau'*Xu2)' ); % *2
    
    g(:,k) =  dEx + dS2;
    g = g + s*bsxfun(@times,pvec,(dPk1minusPk'*X)')...
        + a*s*bsxfun(@times,pvec,(dPuk1minusPuk'*Xu)');
    
    tnll = tnll + nlldet;
end
g=g(:);