function [se, g, yhat] = Linear2NNDet(W,X,k,decode,params,y)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

[W1, b1, W2, b2, W3, b3] = stack2param(W, decode);
[n,p] = size(X);
numh = length(b1);
ps = params.p;
numsample = params.numsamples;

ka = 0.125*pi;
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

mu3 = bsxfun(@plus, ps*Mu2*W3, b3);
s3 = alpha * ps*(1-ps)*(Mu2.*Mu2)*(W3.*W3);
yhat = mu3;

if nargout > 2
    se = 0;
    g = 0;
elseif nargout == 2
    
    se = sum((y(:)-yhat(:)).^2 + s3.^2);
    dmu3 = 2*(yhat - y);
    ds3 = 2*s3;

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
    se = sum((y(:)-yhat(:)).^2 + s3.^2);
end

