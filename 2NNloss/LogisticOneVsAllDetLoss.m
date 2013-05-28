function [nll,g, Mu3, labels] = LogisticOneVsAllDetLoss(W,X,k,decode, params,y1ofn)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

[W1, b1, W2, b2, W3, b3] = stack2param(W, decode);
[n,p] = size(X);
numh = length(b1);
ps = params.p;


ka = 0.125*pi;
% 
% 
% r2 = sqrt(2);
% a = 4-2*r2; b = -log(r2-1);
alpha = 1;
s1 = alpha * ps*(1-ps)*(X.*X)*(W1.*W1);
c1 = 1./sqrt(1+ka*s1);
mu1 = bsxfun(@plus, ps*X*W1, b1);
Mu1 = sigmoid(mu1.*c1);

s2 = alpha * ps*(1-ps)*(Mu1.*Mu1)*(W2.*W2);
c2 = 1./sqrt(1+ka*s2);
mu2 = bsxfun(@plus, ps*Mu1*W2, b2);
Mu2= sigmoid(mu2.*c2);

s3 = alpha * ps*(1-ps)*(Mu2.*Mu2)*(W3.*W3);
c3 = 1./sqrt(1+ka*s3);
mu3 = bsxfun(@plus, ps*Mu2*W3, b3);
Mu3 = sigmoid(mu3.*c3);

% Hw = bsxfun(@plus, Mu2*W3, b3);
% Hwn = bsxfun(@minus, Hw, max(Hw,[],2));
% expHw = exp( Hwn );
% Z = sum(expHw,2);
% expHwdZ = bsxfun(@rdivide, expHw, Z);



if nargout > 2
    [~, labels] = max(Mu3,[], 2);
    nll = 0;
    g = 0;
elseif nargout == 2
    nll = -sum(sum(log(Mu3)./c3.*y1ofn, 2)+sum(log(1-Mu3)./c3.*(1-y1ofn), 2));
    dmu3 = -(y1ofn - Mu3);
    ds3 = alpha *0.5*ka*c3.* ( (y1ofn - Mu3).*mu3.*c3 ...
    - log(Mu3).*y1ofn - log(1-Mu3).*(1-y1ofn) );
    dW3 = ps*(dmu3'*Mu2)' + 2*(ps*(1-ps)) * (ds3'* (Mu2.*Mu2) )'.*W3;
    db3 = sum(dmu3,1);
    
    dMu2 = ds3 * (2*ps*(1-ps)*(W3.*W3))'.*Mu2 + ps * dmu3 * W3';
    dmu2 = dMu2.*Mu2.*(1-Mu2).*c2;
    db2 = sum(dmu2,1);
    ds2 = -alpha *0.5*ka*dMu2.*Mu2.*(1-Mu2).*c2.^3;
    dW2 = ps*(dmu2'*Mu1)' + 2*(ps*(1-ps)) * (ds2'* (Mu1.*Mu1) )'.*W2;
    
    dMu1 = ds2 * (2*ps*(1-ps)*(W2.*W2))'.*Mu1 + ps * dmu2 * W2';
    dmu1 =  dMu1.*Mu1.*(1-Mu1).*c1;
    db1 = sum(dmu1,1);
    ds1 = -alpha * 0.5*ka*dMu1.*Mu1.*(1-Mu1).*c1.^3;
    dW1 = ps*(dmu1'*X)' + 2*(ps*(1-ps)) * (ds1'* (X.*X) )'.*W1;
    
    g = param2stack(dW1, db1, dW2, db2, dW3, db3);
else
    nll = -sum(sum(log(Mu3)./c3.*y1ofn, 2)+sum(log(1-Mu3)./c3.*(1-y1ofn), 2));
end

