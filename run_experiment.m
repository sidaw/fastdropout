addpath(genpath('binaryLRloss'));
load('example_data.mat');
rand('seed', 0)
C = cvpartition(y, 'kfold',2);
%X = bsxfun(@rdivide, X, mean(X,1)+1e-10 );
Xtest = X(C.test(1),:);
ytest = y(C.test(1));
Xtrain = X(C.test(2),:);
ytrain = y(C.test(2));
%%
w_init = 0*randn(size(X,2),1);
mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-2;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 250;
mfOptions.DerivativeCheck = 0;
results = containers.Map;
casenames = {'LR','DetDropout', 'DetDropoutApprox', 'Dropout'};
for casenum = 1:length(casenames)
    obj = casenames{casenum};
    switch obj
        case 'LR'
            funObj = @(w)LogisticLoss(w,Xtrain,ytrain);
            lambdaL2=0.01; 
% you can optimize this value on the test set,
% and LR would still be quite a bit worse
            
        case 'DetDropout'
            funObj = @(w)LogisticLossDetObjDropout(w,Xtrain,ytrain,0.5);
            lambdaL2=0.01;
            
        case 'DetDropoutApprox'
            funObj = @(w)LogisticLossDetObjDropoutDeltaApprox(w,Xtrain,ytrain,0.5);
            lambdaL2=0.01;
            
        case 'Dropout'
            funObj = @(w)LogisticLossMCDropoutSample(w,Xtrain,ytrain,0.5,100,100);
            lambdaL2=0.01;
    end
    
    funObjL2 = @(w)penalizedL2(w,funObj,lambdaL2);
    w = minFunc(funObjL2,w_init,mfOptions);
    ypred = Xtest * w > 0;
    acc = sum(ypred == (ytest+1)/2 )/length(ytest);
    
%     ypred = Xtrain * w > 0;
%     acc = sum(ypred == (ytrain+1)/2 )/length(ytrain);

    
    resultname = [casenames{casenum}];
    results(resultname) = acc;
end

keys = results.keys;
for i=1:length(keys)
    fprintf('%s: %f\n', keys{i}, results(keys{i}));
end
