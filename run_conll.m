addpath(genpath('binaryLRloss'));
addpath(genpath('softmaxLoss'));
addpath(genpath('utils'));

% example or 20newsbydate conll
[Xtrain, ytrain, Xtest, ytest] = getData('conll');
D = size(Xtrain,2);
K = size(ytrain,2);
if ~exist('isconll','var')
    isconll=1;
    evaltrain = 1;
    disp('not defining conll');
end

%%
w_init = 0.01*randn(D*K,1);
mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-3;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 250;
mfOptions.DerivativeCheck = 'off';
testresults = containers.Map;
trainresults = containers.Map;
casenames = {'Softmax', 'SoftmaxDelta'};
% casenames = {'LROnevallDelta','SoftmaxDelta', 'Softmax'};
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
            lambdaL2=0.01;
            
        case 'SoftmaxDeltaCheck'
            funObj = @(w)SoftmaxLossDetObjDropoutDeltaGradTest(w,Xtrain,ytrain,0.5);
            lambdaL2=0.01;
            
        case 'Softmax'
            funObj = @(w)SoftmaxLossFast(w,Xtrain,ytrain);
            lambdaL2=0.1;
            
        end
    lambdaL2=0.001;
    funObjL2 = @(w)penalizedL2(w,funObj,lambdaL2);
    W = minFunc(funObjL2,w_init,mfOptions);
    W = reshape(W, D, K);
    
    resultname = [casenames{casenum}];
    
    save(['W-' resultname], 'W');
    
    ypredsoft = Xtest * W;
    [~, ypredtst] = max(ypredsoft, [], 2);
    acc = sum(ypredtst==oneofktoscalar(ytest)) / size(ytest,1);
    testresults(resultname) = mean(acc);

    ypredsoft = Xtrain * W;
    [~, ypredtr] = max(ypredsoft, [], 2);
    acc = sum(ypredtr == oneofktoscalar(ytrain) ) / size(ytrain,1);
    trainresults(resultname) = mean(acc);
    
    if isconll
        save([resultname '.testres'], 'ypredtst', '-ascii');
        lblcmd = ['echo ' resultname '>> tstresults'];
        pycommandtst = ...
            ['./data/conll-ner/generateconnloutput.py '...
            resultname '.testres data/conll-ner/devfields >' resultname '.conlltest'];
        perlcommandtst = ['./data/conll-ner/conlleval.pl <' resultname '.conlltest' '>> tstresults'];
        
        unix([lblcmd ';' pycommandtst ';' perlcommandtst]);
        
        if evaltrain
        save([resultname '.trainres'], 'ypredtr', '-ascii');
        lblcmdtr = ['echo ' resultname '>> ' 'trainresults'];

        pycommandtr = ...
            ['./data/conll-ner/generateconnloutput.py '...
            resultname '.trainres data/conll-ner/trainfields >' resultname '.conlltrain'];
        perlcommandtr = ['./data/conll-ner/conlleval.pl <' resultname '.conlltrain' '>> trainresults'];

        unix([lblcmdtr ';' pycommandtr ';' perlcommandtr]);
        end
    end
 end

keys = testresults.keys;
for i=1:length(keys)
    fprintf('%s: train=%f test=%f\n', keys{i}, trainresults(keys{i}), testresults(keys{i}));
end
