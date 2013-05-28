function [nll,g] = SoftmaxLossDropout(w,X,y1ofn,k)
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
[is,js,vs] = find(X);
ps = 0.5;
g = zeros(p,k);
numiter = 2500;

nlls = zeros(numiter,1);
for i = 1:numiter
    Zs = sparse(is,js,1*(rand(size(vs)) < ps), n, p, length(is));
%     Z(:,1) = 0.5;
    ZX = Zs.*X;
    %w(:,k) = zeros(p,1);
    Xw = ZX*w;
    expXw = exp(Xw);
    Z = sum(expXw,2);
    expXwdZ = bsxfun(@rdivide, expXw, Z);
    nlls(i) = -sum( sum(Xw.*y1ofn, 2) - log(Z));

    if nargout > 1
        g = g - ((y1ofn - expXwdZ)'*X)';
    end
end

g = reshape(g,[p*(k) 1]) / numiter;
nll = mean(nlls);
