function [nll,g] = SoftmaxLossFast(w,X,y1ofn,k)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * 1
% k = nclasses
%
% This is like SoftmaxLoss2, except w is D*C not D*(C-1),
% since we don't assume the  weights for last class are fixed at 0

% This file is from pmtk3.googlecode.com
[n,p] = size(X);

w = reshape(w,[p k]);
%w(:,k) = zeros(p,1);
Xw = X*w;
expXw = exp(Xw);
Z = sum(expXw,2);
expXwdZ = bsxfun(@rdivide, expXw, Z);
nll = -sum( sum(Xw.*y1ofn, 2) - log(Z));

if nargout > 1
    g = -((y1ofn - expXwdZ)'*X)';
    g = reshape(g,[p*(k) 1]);
end

