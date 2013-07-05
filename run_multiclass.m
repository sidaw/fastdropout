addpath(genpath('binaryLRloss'));
addpath(genpath('softmaxLoss'));
addpath(genpath('utils'));


%%
mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-3;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 150;
mfOptions.DerivativeCheck = 'off';
mfOptions.Corr = 8;
testresults = containers.Map;
trainresults = containers.Map;


if ~exist('params', 'var') || isfield(params, 'reset') && params.reset
    params.discoef = 1;
    params.lambdaL2=0.001;
    params.evaliid = 0;
    params.dataset = 'sector';
    params.isdev = 0;
    params.reset = 0;
    testresults = containers.Map;
    trainresults = containers.Map;
    
    disp('resetting');
end

% example or 20newsbydate conll 20newslibsvm rcv1
datasetname = params.dataset;
[Xtrain, ytrain, Xtest, ytest, Xu] = getData(datasetname);
D = size(Xtrain,2);
K = size(ytrain,2);
w_init = 0.01*randn(D*K,1);


casenames = {'Softmax','SoftmaxDeltaMore', 'SoftmaxDelta', 'LROnevall'}%, 'LROnevallDelta', 'LROnevallDeltaMore', 'LROnevallDet'};
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
    
    ypredsoft = Xtest * W;
    [~, ypredtst] = max(ypredsoft, [], 2);
    acc = sum(ypredtst==oneofktoscalar(ytest)) / size(ytest,1);
    testresults(resultname) = mean(acc);

    ypredsoft = Xtrain * W;
    [~, ypredtr] = max(ypredsoft, [], 2);
    acc = sum(ypredtr == oneofktoscalar(ytrain) ) / size(ytrain,1);
    trainresults(resultname) = mean(acc);
    
    paramstring = sprintf('%s: a=%f,lambda=%f,trainsize=%d', datasetname, params.discoef, params.lambdaL2, size(Xtrain,1) );
    outfile = fopen('multiclassres', 'a+');
    fprintf(outfile, '%s:%s\n', datestr(now), paramstring);
    fprintf(outfile, '%s %s: train=%f test=%f\n', datestr(now), obj, trainresults(resultname), testresults(resultname));
    fclose(outfile);


 end

% keys = testresults.keys;
% paramstring = sprintf('%s: a=%f,lambda=%f,trainsize=%d', datasetname, params.discoef, params.lambdaL2, size(Xtrain,1) );
% outfile = fopen('multiclassres', 'a+');
% fprintf(outfile, '%s:%s\n', datestr(now), paramstring);
% for i=1:length(keys)
%     fprintf(outfile, '%s %s: train=%f test=%f\n', datestr(now), keys{i}, trainresults(keys{i}), testresults(keys{i}));
% end
% fclose(outfile);
