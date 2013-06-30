function [nll,g] = SoftmaxLossFast(w,X,y1ofn,k)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses
if nargin<4
    k = size(y1ofn,2);
end

[n,p] = size(X);

w = reshape(w,[p k]);
%w(:,k) = zeros(p,1);
Xw = X*w;
Xw = bsxfun(@minus, Xw, max(Xw,[],2) );
expXw = exp(Xw);
Z = sum(expXw,2);
expXwdZ = bsxfun(@rdivide, expXw, Z);
nll = -sum( sum(Xw.*y1ofn, 2) - log(Z));

if nargout > 1
    g = -((y1ofn - expXwdZ)'*X)';
    g = reshape(g,[p*(k) 1]);
end

