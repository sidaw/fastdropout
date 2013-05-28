function [nll,g,H] = LogisticLossMCDropoutSample(w,X,y,ps, miniter, numiter)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
[is,js,vs] = find(X);

% numiter = 10;
% miniter = 200;
nlls = zeros(numiter,1);
g = zeros(p, 1);
numkeep = 10;
% mystats = zeros(numkeep, numiter);
for i = 1:numiter
    Z = sparse(is,js,1*(rand(size(vs)) < ps), n, p, length(is));
    % comment this line to dropout bias as well
    % Z(:,1) = 0.8;
    ZX = Z.*X;
    ZXw = (ZX)*w;
%     mystats(:, i) = ZXw(1:numkeep);
    yXw = y.*ZXw;
    nlls(i) = sum(mylogsumexp([zeros(n,1) -yXw]));

    g = g + -ZX.'*(y./(1+exp(yXw)));
    
    if i > miniter && (std(nlls(1:i),1)/sqrt(i) < 1e-3*mean(nlls(1:i)) )
        break
    end
    % else
    %     g = -ZX.'*(y./(1+exp(yXw)));
end
g = g ./ numiter;
% fprintf('var of mean nlls %f at i=%d\n', std(nlls(1:i),1)/sqrt(i),i);
nll = mean(nlls(1:i));


