function [nll,g, expHwdZ, labels] = Softmax2NN_MCDropoutMulti(W,X,k,decode,params,y1ofn)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

M = params.numsamples;
nll = 0;
g = zeros(size(W));

% if isempty(y1ofn)
%     y1ofn = zeros(size(X,1), k);
% end

expHwdZ = zeros(size(X,1), k);

for i = 1:M
    [nlli, gi, expHwdZi, ~] = Softmax2NN_MCDropoutSingle(W,X,k,decode,params);
    nll = nll + nlli;
    g = g + gi;
    expHwdZ = expHwdZ + expHwdZi;
end
g = g / M;
nll = nll / M;
expHwdZ = bsxfun(@rdivide, expHwdZ, sum(expHwdZ,2));
if nargout > 2
    [~, labels] = max(expHwdZ,[], 2);
    nll = 0;
    g = 0;
end