addpath(genpath('binaryLRloss'));
 addpath(genpath('utils'));
% load('~/Dropbox/projects/naiveBayes/data/rt10662/bigram_rts.mat');
%load('~/Dropbox/projects/naiveBayes/data/20ng/bigram_ng20_atheisms_strip_noheader.mat');
% load('~/Dropbox/projects/naiveBayes/data/rt2k/unigram_rt2k.mat');
%load('~/Dropbox/projects/naiveBayes/data/rt2k/bigram_rt2k.mat');
% load('~/Dropbox/projects/naiveBayes/data/mrl/unigram_mrl_unsuptok.mat')
% labels = labels(1:50000);
% [Xl, yl] = listToVecs(allSNumBi, labels, wordsbi);
% %%
% %load('example_data.mat');
%rand('seed', 6)
s = RandStream('swb2712','Seed',0);
RandStream.setDefaultStream(s);


Xtest = Xl(25001:end,:);
ytest = yl(25001:end);

ytrainAll = yl(1:25000);
XtrainAll = Xl(1:25000, :);


N = length(ytrainAll);
[~, sortind] = sort(ytrainAll);
interleave = [1:N; N:-1:1];
interleave = reshape(interleave(:, 1:end/2), 1, []);
sortedfinalind = sortind(interleave);
XtrainAll = XtrainAll(sortedfinalind, :);
ytrainAll = ytrainAll(sortedfinalind);


mfOptions.Method = 'lbfgs';
mfOptions.optTol = 2e-2;
mfOptions.progTol = 2e-6;
mfOptions.LS = 2;
mfOptions.LS_init = 2;
mfOptions.MaxIter = 40;
mfOptions.DerivativeCheck = 0;
results = containers.Map;
% l2set = [1e-5, 1e-4, 1e-3, 1e-2:0.02:0.08, 1e-1:0.1:0.9,1:5,10];
datasubs = [0.002 0.004 0.008];% 0.02:0.02:0.08 0.15 0.3 0.7 1];
%datasubs = [0.2 1];
accs = []

casenames = {'LR', 'NaiveBayes', 'DetDropout-0.2', 'DetDropout-0.5', 'DetDropout-0.8'};%, 'DetDropout-0.5', 'DetDropout-0.8'};
casenames = {'LRsgd', 'NaiveBayes', 'Dropout-0.2', 'Dropout-0.5', 'Dropout-0.8'};%, 'DetDropout-0.5', 'DetDropout-0.8'};
casenames = {'LR', 'NaiveBayes', 'Dropout-0.2', 'Dropout-0.5', 'Dropout-0.8'};%, 'DetDropout-0.5', 'DetDropout-0.8'};
%casenames = {'LRsgd', 'NaiveBayes', 'DetDropout-0.2', 'DetDropout-0.5', 'DetDropout-0.8'};%, 'DetDropout-0.5', 'DetDropout-0.8'};

%casenames = {'LR', 'NaiveBayes', 'DetDropout-0.2', 'DetDropout-0.5', 'DetDropout-0.8'};%, 'DetDropout-0.5', 'DetDropout-0.8'};
%casenames = {'Dropout-0.2', 'Dropout-0.8'}
for datai = 1:length(datasubs)
    
    dataend = ceil(datasubs(datai)*length(ytrainAll));
    
    Xtrain = XtrainAll(1:dataend, :);
    ytrain = ytrainAll(1:dataend);
    for casenum = 1:length(casenames)
        obj = casenames{casenum};
        w_init = 0*randn(size(XtrainAll,2),1);
        
        dashind = strfind(obj, '-');
        pstay = 0;
        if dashind>0
            method = obj(1:dashind-1);
            pstay = str2double(obj(dashind+1:end));
        else
            method = obj;
        end
        
        switch method
            case 'LR'
                funObj = @(w)LogisticLoss(w,Xtrain,ytrain);
                lambdaL2=0.01;
            case 'LRsgd'
                funObj = @(w,X,y)LogisticLoss(w,X,y);
                lambdaL2=0.01;
                % you can optimize this value on the test set,
                % and LR would still be quite a bit worse
                
            case 'DetDropout'
                funObj = @(w)LogisticLossDetObjDropout(w,Xtrain,ytrain,pstay);
                lambdaL2=0.01;
                
            case 'Dropout'
                funObj = @(w,X,y)LogisticLossMCDropout(w,X,y,pstay);
                lambdaL2=0.01;
                
                            
            case 'NaiveBayes'
                [wnb] = trainMNB(Xtrain, ytrain, 1);
                w = wnb;
        end
        lambdaL2 = 0.001;
        if ~strcmp(method, 'NaiveBayes')
            
            if strcmp(method, 'Dropout') | strcmp(method, 'LRsgd')
                options.MaxIter = 500;
                if pstay < 0.5
                    options.MaxIter = 800;
                end
                options.BatchSize = 25000;
                options.eta = 1e-1;
                funObjL2 = @(w,X,y) penalizedL2(w,@(w)funObj(w,X,y),lambdaL2);
                w = minFuncAdagrad(funObjL2,w_init, Xtrain, ytrain, options);
            else
                lambdaL2 = 0.00;
                funObjL2 = @(w)penalizedL2(w,funObj,lambdaL2);
                w = minFunc(funObjL2,w_init,mfOptions);
            end
        end
        softpred = Xtest * w;
        softpredn = softpred - mean(softpred);
        ypred = softpredn > 0;
        acc = sum(ypred == (ytest+1)/2 )/length(ytest);
        accs(datai, casenum) = acc;
        %     ypred = Xtrain * w > 0;
        %     acc = sum(ypred == (ytrain+1)/2 )/length(ytrain);
    end
end

%%
close all

hfig=figure(1);
hold on
datasubsn = datasubs * length(ytrainAll);
semilogx(datasubsn, 100-100*accs(:,1), 'r-', 'LineWidth', 1.5)

semilogx(datasubsn, 100-100*accs(:,2), 'k-', 'LineWidth', 1.5)
semilogx(datasubsn, 100-100*accs(:,3), '--', 'Color', [0.2 0 0], 'LineWidth', 1)
semilogx(datasubsn, 100-100*accs(:,4), '--', 'Color', [0.5 0 0], 'LineWidth', 1)
semilogx(datasubsn, 100-100*accs(:,5), '--', 'Color', [0.8 0 0], 'LineWidth', 1)
hold off
xlabel('n', 'Interpreter','tex');
ylabel('Test Error Rate (%)', 'Interpreter','tex');
l=legend('Log.Reg.','Naive Bayes', 'Dropout-0.8', 'Dropout-0.5', 'Dropout-0.2','Location','NorthEast');
legend boxoff
set(hfig,'Position',[100 200 300 250]);

export_fig perf_vs_data_imdb.pdf -pdf -transparent


