function [nll,g, expHwdZ, labels] = Softmax2NN(W,X,k,decode,params,y1ofn)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

[W1, b1, W2, b2, W3, b3] = stack2param(W, decode);
[n,p] = size(X);

mu1 = bsxfun(@plus, X*W1, b1);
Mu1 = sigmoid(mu1);

mu2 = bsxfun(@plus, Mu1*W2, b2);
Mu2= sigmoid(mu2);

Hw = bsxfun(@plus, Mu2*W3, b3);
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
    dW3 = -((y1ofn - expHwdZ)'*Mu2)';
    db3 = -sum(y1ofn - expHwdZ,1);
    
    dMu2 = -(y1ofn - expHwdZ) * W3' ;
    dmu2 = dMu2.*Mu2.*(1-Mu2);
    db2 = sum(dmu2,1);
    dW2 = (dmu2'*Mu1)';
    
    dMu1 = dmu2 * W2';
    dmu1 =  dMu1.*Mu1.*(1-Mu1);
    db1 = sum(dmu1,1);
    dW1 = (dmu1'*X)';
    g = param2stack(dW1, db1, dW2, db2, dW3, db3);
else
    nll = -sum( sum(Hwn.*y1ofn, 2) - log(Z));
end

