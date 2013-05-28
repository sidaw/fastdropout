function [nll,g] = LogisticLossMCDropoutSampleGaussIntegrate(w,X,y,ps, numsample)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

[n,p] = size(X);
pvec = ps * ones(p, 1);
% uncomment next line to never drop out the bias
pvec(1) = 0.9;
EXw = X*(w .* pvec);
SigmaXw2 = (X.*X) * (w.*w.*pvec.*(1-pvec));
SigmaXw = sqrt(SigmaXw2);
% yEXw = y .* EXw;
% randn('seed', 5)
sigmar =  bsxfun(@times, SigmaXw, randn(n, numsample));
Xw = bsxfun(@plus, EXw, sigmar);

yXw =  bsxfun(@times, Xw, y);
expyXw = exp(yXw);
alpha = mean(bsxfun(@times, 1./(1+expyXw), y), 2);

beta = mean(bsxfun(@times, sigmar./(1+expyXw), y), 2) ./ SigmaXw2;
changewp = (w.*(1-pvec))';
meandiff = bsxfun(@times, X,  changewp);

% batchsize = 5000;
% for i = 1:ceil(n/batchsize)
%     batchstart = (i-1)*batchsize + 1;
%     batchend = min(i*batchsize,n);
%     batch = batchstart:batchend;
%     meandiff(i,batch) = bsxfun(@times, X(i,batch), changewp);
% end
meandiff = bsxfun(@times, meandiff', beta');

% gamma = mean(bsxfun(@times, sigmar.*sigmar./(1+expyXw), y), 2) ./ (2.*SigmaXw2.*SigmaXw2);
% gamma = gamma + 0.5./SigmaXw2;
% sigmadiff = bsxfun(@times, X.*X, (w.*w.*(1-pvec).*(pvec))' );
% sigmadiff = bsxfun(@times, sigmadiff', gamma');
% - sum(X.' .* sigmadiff,2)
g = -pvec .* ( X.'*alpha+sum(X.' .* meandiff,2) );

nll = sum(mean(log(1+1./expyXw), 2));
% nll = sum(mean( log1plusexp(-yXw), 2));
