addpath(genpath('binaryLRloss'));
addpath(genpath('softmaxLoss'));
addpath(genpath('utils'));



%%

mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-3;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 250;
mfOptions.DerivativeCheck = 'off';

doreset = isfield(params, 'reset') && params.reset;
if ~exist('params', 'var') || doreset
params.discoef = 0.1;
params.lambdaL2=0.001;
params.evaliid = 0;
params.dataset = 'conll';
params.isdev = 0;
params.reset = 0;
testresults = containers.Map;
trainresults = containers.Map;
if reset
    disp('resetting');
end
end

disp(mfOptions)
disp(params)

% example or 20newsbydate conll
[Xtrain, ytrain, Xtest, ytest, Xu] = getData(params.dataset);
D = size(Xtrain,2);
K = size(ytrain,2);
w_init = 0.01*randn(D*K,1);

casenames = {'SoftmaxDeltaMore', 'SoftmaxDelta', ...
    'Softmax'};
% ,...    'LROnevall', 'LROnevallDelta'};
% casenames = { ...
%    'LROnevallDeltaMore', 'LROnevallDet'};
for casenum = 1:length(casenames)
    obj = casenames{casenum};
    switch obj
        case 'LROnevall'
            funObj = @(w)LogisticOnevsAllLoss(w,Xtrain,ytrain);
        case 'LROnevallDelta'
            funObj = @(w)LogisticOnevsAllLossDetObjDropoutDelta(w,Xtrain,ytrain,0.5);
        case 'LROnevallDeltaMore'
            funObj = @(w)LogisticOnevsAllLossDetObjDropoutDeltaMoreData(w,Xtrain,ytrain,0.5, Xu, params.discoef);
        case 'LROnevallDet'
            funObj = @(w)LogisticOnevsAllLossDetObjDropout(w,Xtrain,ytrain,0.5);

            
        case 'SoftmaxDelta'
            funObj = @(w)SoftmaxLossDetObjDropoutDelta(w,Xtrain,ytrain,0.5);            
        case 'SoftmaxDeltaMore'
            funObj = @(w)SoftmaxLossDetObjDropoutDeltaMoreData(w,Xtrain,ytrain,0.5, Xu, params.discoef);
        case 'Softmax'
            funObj = @(w)SoftmaxLossFast(w,Xtrain,ytrain);
            
        case 'SoftmaxDeltaCheck'
            funObj = @(w)SoftmaxLossDetObjDropoutDeltaGradTest(w,Xtrain,ytrain,0.5);

    end
    funObjL2 = @(w)penalizedL2(w,funObj,params.lambdaL2);
    W = minFunc(funObjL2,w_init,mfOptions);
    W = reshape(W, D, K);
    
    resultname = [casenames{casenum}];
    
    save(['W-' resultname], 'W');
    
    paramstring = sprintf('%s:a=%f,lambda=%f,trainsize=%d', resultname, params.discoef, params.lambdaL2, size(Xtrain,1) );
    % using actual CoNLL scheme, not using iid data

    ypredsoft = Xtest * W;
    [~, ypredtst] = max(ypredsoft, [], 2);
    acc = evalconll(params.evaliid, ypredtst, ytest, resultname, paramstring, 'conlltest', params.isdev);
    testresults(resultname) = acc;

    ypredsoft = Xtrain * W;
    [~, ypredtr] = max(ypredsoft, [], 2);
    acc = evalconll(params.evaliid, ypredtr, ytrain, resultname, paramstring, 'conlltrain', params.isdev);
    trainresults(resultname) = acc;
    
        
 end

keys = testresults.keys;
for i=1:length(keys)
    fprintf('%s: train=%f test=%f\n', keys{i}, trainresults(keys{i}), testresults(keys{i}));
end
