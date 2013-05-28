function [nll,g, expHwdZ, labels] = SoftmaxClassification(W,X,k,decode,params,y1ofn)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

[W3, b3] = stack2param(W, decode);
[n,p] = size(X);
Hw = bsxfun(@plus, X*W3, b3);
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
    dW3 = -((y1ofn - expHwdZ)'*X)';
    db3 = -sum(y1ofn - expHwdZ,1);
    g = param2stack(dW3, db3);
else
    nll = -sum( sum(Hwn.*y1ofn, 2) - log(Z));
end

