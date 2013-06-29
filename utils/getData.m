function [Xtrain, ytrain, Xtest, ytest] = getData(dataname)
 
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
        Xtestind = load('data/20news-bydate/matlab/test.data');
        Xtest=spconvert(Xtestind);
        Xtrainind = load('data/20news-bydate/matlab/train.data');
        Xtrain=spconvert(Xtrainind);
        
        ytestind = load('data/20news-bydate/matlab/test.label');
        ytrainind = load('data/20news-bydate/matlab/train.label');
        
        ytest=to1ofk(ytestind, 20);
        ytrain=to1ofk(ytrainind, 20);
        
end