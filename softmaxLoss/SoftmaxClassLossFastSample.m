function [nll,g, EexpdZ, labels] = SoftmaxClassLossFastSample(W,X,k,decode,params,y1ofn)
% w is nfeatures * nclasses 
% X is ncases * nfeatures
% y is ncases * nclasses
% k = nclasses

[W3, b3] = stack2param(W, decode);
[n,p] = size(X);
ps = params.p;
ps1 = params.p1;
% if params.weighthalving == 1
%     W1 = W1*ps1;
%     b1 = b1*ps1;
%     
%     W2 = W2*ps;
%     b2 = b2*ps;
%     W3 = W3*ps;
%     b3 = b3*ps;
% end
numsample = params.numsamples;

ka = 0.125*pi;
% 
% 
% r2 = sqrt(2);
% a = 4-2*r2; b = -log(r2-1);
alpha = 1;

mu3 = bsxfun(@plus, ps*X*W3, b3);
s3 = alpha * ps*(1-ps)*(X.*X)*(W3.*W3);
mu3n = bsxfun(@minus, mu3, max(mu3,[],2));

% disp('FIXING SEED!')
% randn('seed', 0)

stdnoise = randn([size(s3), numsample]);
z = bsxfun(@times, stdnoise, sqrt(s3));
samples = bsxfun(@plus, mu3n, z);


expsamples = exp( samples );
Zsamples = sum(expsamples,2);
expsdZ = bsxfun(@rdivide, expsamples, Zsamples);
EexpdZ = mean(expsdZ,3);
if nargout > 3
    Eexpsdz = mean(expsdZ,3);
    [~, labels] = max(Eexpsdz,[], 2);
    nll = 0;
    g = 0;
elseif nargout > 1
    tlogp = bsxfun(@times, log(expsdZ+eps), y1ofn);
    nll = -sum(sum(mean( tlogp, 3)));
    dmu3 = -(y1ofn - mean(expsdZ,3));
    ds3 = - (mean(bsxfun(@times, z, y1ofn)...
              - bsxfun(@times,z,expsdZ),3));

    dW3 = ps*(dmu3'*X)' + 2*(ps*(1-ps)) * (ds3'* (X.*X) )'.*W3;
    db3 = sum(dmu3,1);
    

    g = param2stack(dW3, db3);
else
    tlogp = bsxfun(@times, log(expsdZ+eps), y1ofn);
    nll = -sum(sum(mean( tlogp, 3)));
end

