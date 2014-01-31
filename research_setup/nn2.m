addpath(genpath('lossfuncs'));
% addpath(genpath('minFunc_2012'));
if ~exist('hyperparams', 'var')
    disp('Using default, supply hyperparams to set your own');
    gethyperparams
end
disp(hyperparams)
Ntrain = hyperparams.Ntrain;
numh2 = hyperparams.numh2;
numh1 = hyperparams.numh1;
lambda = hyperparams.lambda; 
lossfunc = hyperparams.lossfunc;
nclass = 10;
% setSeed(0);
if hyperparams.dataset == 0
[Xtrain, ytrain, Xtest, ytest] = mnistLoad([0:nclass-1], Ntrain);
Xtrain = Xtrain/255;
Xtest = Xtest/255;
elseif hyperparams.dataset == 1
[Xtrain, ytrain, Xtest, ytest] = cifarLoad([0:nclass-1], Ntrain);
Xtrain = Xtrain/255;
Xtest = Xtest/255;
end

ytrain = setSupport(ytrain, [1:nclass]);
ytest = setSupport(ytest, [1:nclass]);
[N,D] = size(Xtrain);
winit = zeros(D*nclass,1); % randn(D,1);
%funObj = @(w)LogisticLossScaled(w,Xtrain,ytrain);
[n,p] = size(Xtrain);
k = nclass;
y1ofn = zeros(n, k);
y1ofn(sub2ind([n,k], 1:n, ytrain'))=1;

numv = p;

W1 = hyperparams.initmulti*randn(numv,numh1);
b1 = zeros(1, numh1);
W2 = hyperparams.initmulti*randn(numh1,numh2);
b2 = zeros(1, numh2);
W3 = hyperparams.initmulti*randn(numh2,nclass);
b3 = zeros(1, nclass);

[W params] = param2stack(W1, b1, W2, b2, W3, b3);
if exist('useprev', 'var') && useprev && exist('w', 'var')
  disp('trying to use previous params')
  W = w;
end
funObj = @(w) penalizedL2(w, @(ww) lossfunc(ww, Xtrain, nclass, params, hyperparams, y1ofn), lambda);
% W = w;
if ~exist('options', 'var')
    disp('usiong default options')
    getoptions
end
disp(options);

% methods = {'sd', 'cg', 'bb', 'lbfgs'};
methods = {'lbfgs'};


if isfield(hyperparams, 'numiter')
    numiter = hyperparams.numiter;
else
    numiter = 5;
end


for k = 1:numiter
  tic;
%   funObj = @(w, Xtrain, y1ofn) penalizedL2(w, @(ww) lossfunc(ww, Xtrain, nclass, params, hyperparams, y1ofn), lambda);
%   [w, finalObj, exitflag] = minFuncsd(funObj, Xtrain, y1ofn, W, options);

  [w, finalObj, exitflag] = minFunc(funObj, W, options);
  W = w;
  nclass = size(y1ofn,2);
  [~, ~, expHwdZ, trlabels] = lossfunc(w, Xtrain, nclass,params, hyperparams, []);
  [~, ~,expHwdZ, tslabels] = lossfunc(w, Xtest, nclass,params, hyperparams, []);
  t1 = toc;
  disp(t1)
  tacc = sum(tslabels == ytest) ./ length(ytest)
  tracc = sum(trlabels == ytrain) ./ length(ytrain)
  writeResults(tacc, tracc, hyperparams)
  save(['results/nn2_' num2str(k) '_' getparamsstring(hyperparams) '.mat'], 'w', 'tacc', 'tracc', 'params','hyperparams');
  end
  if 0
     figure;
     plot(fvalTrace, 'o-', 'linewidth', 2);
     title(sprintf('%s, %5.3f seconds, final obj = %5.3f acc = %5.3f', ...
        method, t, finalObj, accuracy));
      printPmtkFigure(sprintf('logregOpt%s', method))
     [W1, b1, W2, b2,W3, b3] = stack2param(w, params);
     for i = 1:10
     subplot(1,10,i);
     imagesc(reshape(W1(:,i), 28, 28));
     colormap(gray)
     end
  end

