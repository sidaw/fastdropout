addpath(genpath('binaryLRloss'));
addpath(genpath('softmaxLoss'));
addpath(genpath('utils'));

% example or 20newsbydate conll
[Xtrain, ytrain, Xtest, ytest] = getData('conll');
D = size(Xtrain,2);
K = size(ytrain,2);

%%
w_init = 0.01*randn(D*K,1);
mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-3;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 250;
mfOptions.DerivativeCheck = 'off';



if ~exist('params', 'var')
params.discoef = 0.1;
params.lambdaL2=0.001;
params.evaliid = 0;
testresults = containers.Map;
trainresults = containers.Map;
end

disp(mfOptions)
disp(params)

casenames = {'SoftmaxDeltaMore', 'SoftmaxDelta', ...
    'Softmax',...
    'LROnevall', 'LROnevallDelta'};
casenames = { ...
    'LROnevallDeltaMore', 'LROnevallDet'};
Xu = Xtest;
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
    
    ypredsoft = Xtest * W;
    [~, ypredtst] = max(ypredsoft, [], 2);
    acc = sum(ypredtst==oneofktoscalar(ytest)) / size(ytest,1);
    testresults(resultname) = mean(acc);

    ypredsoft = Xtrain * W;
    [~, ypredtr] = max(ypredsoft, [], 2);
    acc = sum(ypredtr == oneofktoscalar(ytrain) ) / size(ytrain,1);
    trainresults(resultname) = mean(acc);
    paramstring = sprintf('%s:a=%f,lambda=%f,trainsize=%d', resultname, params.discoef, params.lambdaL2, size(Xtrain,1) );
    % using actual CoNLL scheme, not using iid data
    if params.evaliid == 0
        save([resultname '.testres'], 'ypredtst', '-ascii');
        lblcmd = ['echo ' paramstring '>> tstresults'];
        pycommandtst = ...
            ['./data/conll-ner/generateconnloutput.py '...
            resultname '.testres data/conll-ner/devfields >' resultname '.conlltest'];
        perlcommandtst = ['./data/conll-ner/conlleval.pl <' resultname '.conlltest' '>> tstresults'];
        
        unix([lblcmd ';' pycommandtst ';' perlcommandtst]);
        
        if 1
        save([resultname '.trainres'], 'ypredtr', '-ascii');
        lblcmdtr = ['echo ' paramstring '>> ' 'trainresults'];

        pycommandtr = ...
            ['./data/conll-ner/generateconnloutput.py '...
            resultname '.trainres data/conll-ner/trainfields >' resultname '.conlltrain'];
        perlcommandtr = ['./data/conll-ner/conlleval.pl <' resultname '.conlltrain' '>> trainresults'];

        unix([lblcmdtr ';' pycommandtr ';' perlcommandtr]);
        end
    else
        % tokenmap = ['B-LOC', 'B-MISC', 'B-ORG', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O'];
        ytesttrue = oneofktoscalar(ytest);
        fname = [resultname '.tst.idd'];
        file = fopen(fname, 'w+');
        for j = 1:length(ytesttrue)
            fprintf(file, 'dontknowtoken %d %d\n', ytesttrue(j), ypredtst(j) );
        end
        perlcommandtst = ['./data/conll-ner/conlleval.pl -r -o 8 < ' fname '>> tstresults'];
        lblcmd = ['echo idd' paramstring '>> tstresults'];
        unix([lblcmd ';' perlcommandtst]);
        
        
        ytraintrue = oneofktoscalar(ytrain);
        fname = [resultname '.tr.idd'];
        file = fopen(fname, 'w+');
        for j = 1:length(ytraintrue)
            fprintf(file, 'dontknowtoken %d %d\n', ytraintrue(j), ypredtr(j) );
        end
        perlcommandtr = ['./data/conll-ner/conlleval.pl -r -o 8 < ' fname '>> trainresults'];
        lblcmd = ['echo idd' paramstring '>> tstresults'];
        unix([lblcmd ';' perlcommandtr]);
    end
 end

keys = testresults.keys;
for i=1:length(keys)
    fprintf('%s: train=%f test=%f\n', keys{i}, trainresults(keys{i}), testresults(keys{i}));
end
