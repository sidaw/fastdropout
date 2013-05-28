function [nll,g,H] = LogisticLossMCDropout(w,X,y,ps)
% Monte Carlo real dropout, single sample,
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
[is,js,vs] = find(X);
Z = sparse(is,js,1*(rand(size(vs)) < ps), n, p, length(is));
Z(:,1) = 1;
ZX = Z.*X;
Xw = (ZX)*w;
yXw = y.*Xw;

nll = sum(mylogsumexp([zeros(n,1) -yXw]));

if nargout > 2
    sig = 1./(1+exp(-yXw));
    g = -ZX.'*(y.*(1-sig));
else
    g = -ZX.'*(y./(1+exp(yXw)));
end