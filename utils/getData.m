function [Xtrain, ytrain, Xtest, ytest] = getData(dataname, filter)

if ~exist('filter','var')
    filter = 100;
end

switch dataname
    case 'example' 
        load('example_data.mat');
        rand('seed', 0)
        C = cvpartition(y, 'kfold',2);
        y=(y+1)/2;
        y=to1ofk(y,2);
        Xtest = X(C.test(1),:);
        ytest = y(C.test(1),:);
        Xtrain = X(C.test(2),:);
        ytrain = y(C.test(2),:);
    case '20newsbydate'
        ytestind = load('data/20news-bydate/matlab/test.label');
        ytrainind = load('data/20news-bydate/matlab/train.label');
        
        ytest=to1ofk(ytestind, 20);
        ytrain=to1ofk(ytrainind, 20);

        Xtestind = load('data/20news-bydate/matlab/test.data');
        Xtest=spconvert(Xtestind);
        Xtrainind = load('data/20news-bydate/matlab/train.data');
        Xtrain=sparse(length(ytrainind), size(Xtest,2));
        maxtrainind= max(Xtrainind,[],1);
        Xtrain(:,1:maxtrainind(2))=spconvert(Xtrainind);
        
        Xtrain = [ones(size(Xtrain,1),1), Xtrain];
        Xtest = [ones(size(Xtest,1),1), Xtest];
        
   case 'conll'
%         302811 alllabels
%         204567 trainlabels
%         51578 devlabels
%         46666 testlabels
        trainsize = 204567;
        devsize = 51578;

       
        yind = load('data/conll-ner/alllabels');
      
        y=to1ofk(yind,8);
        Xind = load('data/conll-ner/allvecs');
        X=spconvert(Xind);
        X = [ones(size(X,1),1), X];
        
        ytrain = y(1:trainsize,:);
        Xtrain = X(1:trainsize,:);
        
        ytest = y(trainsize+1:trainsize+devsize,:);
        Xtest = X(trainsize+1:trainsize+devsize,:);
        
        Xtrain = Xtrain(1:10000,:);
        ytrain = ytrain(1:10000,:);
end


if filter<=size(ytrain,2)
ytestfilter = ytestind <= filter;
ytrainfilter = ytrainind <= filter;

Xtrain=1*(Xtrain(ytrainfilter, :)>0);
ytrain=ytrain(ytrainfilter, 1:filter);

Xtest=1*(Xtest(ytestfilter, :)>0);
ytest=ytest(ytestfilter, 1:filter);

end

