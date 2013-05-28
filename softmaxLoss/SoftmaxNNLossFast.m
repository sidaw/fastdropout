function [nll,g, expHwdZ, labels] = SoftmaxNNLossFast(W,X,k,params,y1ofn)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

[W1, b1, W2, b2] = stack2param(W, params);
[n,p] = size(X);
numh = length(b1);
ps = 0.5;

a = 2;
mu1 = bsxfun(@plus, ps*X*W1, b1);
s1 = a*ps*(1-ps)*(X.*X)*(W1.*W1);

ka = 0.125*pi;
c = 1./sqrt(1+ka*s1);

mu2 = sigmoid(mu1.*c);

Hw = bsxfun(@plus, mu2*W2, b2);
Hwn = bsxfun(@minus, Hw, max(Hw,[],2));
expHw = exp( Hwn );
Z = sum(expHw,2);
expHwdZ = bsxfun(@rdivide, expHw, Z);

if nargout > 2
    [~, labels] = max(expHwdZ,[], 2);
    nll = 0;
    g = 0;
elseif nargout == 2
    nll = -sum( sum(Hwn.*y1ofn, 2) - log(Z));
    dW2 = -((y1ofn - expHwdZ)'*mu2)';
    db2 = -sum(y1ofn - expHwdZ,1);
    dmu2 = -(y1ofn - expHwdZ) * W2' ;
    dmu1 = dmu2.*mu2.*(1-mu2).*c;
    db1 = sum(dmu1,1);
    ds1 = -0.5*ka*dmu2.*mu2.*(1-mu2).*c.^3;
    dW1 = ps*(dmu1'*X)' + a*(ps*(1-ps)) * (ds1'* (X.*X) )'.*W1;
    g = param2stack(dW1, db1, dW2, db2);
end

