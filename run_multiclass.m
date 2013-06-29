addpath(genpath('binaryLRloss'));
addpath(genpath('softmaxloss'));
addpath(genpath('utils'));

% example or 20newsbydate
[Xtrain, ytrain, Xtest, ytest] = getData('example');
D = size(Xtrain,2);
K = size(ytrain,2);

%%
w_init = 0.1*randn(D*K,1);
mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-3;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 5;
mfOptions.DerivativeCheck = 1;
testresults = containers.Map;
trainresults = containers.Map;
casenames = {'LROnevall', 'LROnevallDelta', 'SoftmaxDelta'};
for casenum = 1:length(casenames)
    obj = casenames{casenum};
    switch obj
        case 'LROnevall'
            funObj = @(w)LogisticOnevsAllLoss(w,Xtrain,ytrain);
            lambdaL2=0.01; 
            
        case 'LROnevallDelta'
            funObj = @(w)LogisticOnevsAllLossDetObjDropoutDelta(w,Xtrain,ytrain,0.5);
            lambdaL2=0.01;
            
        case 'SoftmaxDelta'
            funObj = @(w)SoftmaxLossDetObjDropoutDelta(w,Xtrain,ytrain,0.5);
            lambdaL2=0.1;
        end
    
    funObjL2 = @(w)penalizedL2(w,funObj,lambdaL2);
    W = minFunc(funObjL2,w_init,mfOptions);
    W = reshape(W, D, K);
    
    resultname = [casenames{casenum}];
    
    ypredsoft = Xtest * W;
    [~, ypred] = max(ypredsoft, [], 2);
    acc = sum(to1ofk(ypred, K) == ytest) / size(ytest,1);
    testresults(resultname) = mean(acc);

    ypredsoft = Xtrain * W;
    [~, ypred] = max(ypredsoft, [], 2);
    acc = sum(to1ofk(ypred, K) == ytrain) / size(ytrain,1);
    trainresults(resultname) = mean(acc);
end

keys = testresults.keys;
for i=1:length(keys)
    fprintf('%s: train=%f test=%f\n', keys{i}, trainresults(keys{i}), testresults(keys{i}));
end
