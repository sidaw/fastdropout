function [Xtrain, ytrain, Xtest, ytest, Xu] = getData(dataname, filter)

if ~exist('filter','var')
    filter = 500;
end
Xu = 0;

switch dataname
    case 'reuters'
        temp = load('data/matfiles/Reuters21578.mat');
        matlabgenereal()
        breakIntoParts();
        
    case 'rcv4'
        temp = load('data/matfiles/RCV1_4Class.mat');
        matlabgenereal()
        %breakIntoParts();
        
    case 'tdt2_top30'
        temp = load('data/matfiles/TDT2.mat');
        matlabgenereal()
        %breakIntoParts();
        
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
    case '20newslibsvm'
        getLIBSVMstd('data/20news-libsvm/news20', 'data/20news-libsvm/news20.t', 1)
    case '20newslibsvmsemi'
        getLIBSVMstd('data/20news-libsvm/news20', 'data/20news-libsvm/news20.t', 1)
        breakIntoParts();
    case 'sectorsemi'
        getLIBSVMstd('data/sectorscale/sector.scale', 'data/sectorscale/sector.t.scale', 1)
        breakIntoParts();
        
        %Xu = X(2*part+1:end,:);
    case 'sector'
        getLIBSVMstd('data/sectorscale/sector.scale', 'data/sectorscale/sector.t.scale', 1)
    case 'sectorscale'
        getLIBSVMstd('data/sectorscale/sector.scale', 'data/sectorscale/sector.t.scale', 0)
    case 'protein'
        getLIBSVMstd('data/protein/protein', 'data/protein/protein.t', 0)
    case 'rcv1'
        getLIBSVMstd('data/rcv1/rcv1_train.multiclass', 'data/rcv1/rcv1_test.multiclass', 1)

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
        
        ytest = y(trainsize+devsize+1:end,:);
        Xtest = X(trainsize+devsize+1:end,:);
        
        Xu = Xtest;
        
        Xtrain = Xtrain(1:10000,:);
        ytrain = ytrain(1:10000,:);
        
    case 'conlliid'
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
        perm = randperm(size(X,1));
        X = X(perm, :);
        y = y(perm, :);
        trsize = 10000;
        tssize = 10000;
        
        ytrain = y(1:trsize,:);
        Xtrain = X(1:trsize,:);
        
        ytest = y(trsize+1:trsize+tssize,:);
        Xtest = X(trsize+1:trsize+tssize,:);
        
        Xu = X(trsize+tssize:trsize+2*tssize,:);
end


if filter<=size(ytrain,2)
    ytestfilter = ytestind <= filter;
    ytrainfilter = ytrainind <= filter;
    
    Xtrain=1*(Xtrain(ytrainfilter, :)>0);
    ytrain=ytrain(ytrainfilter, 1:filter);
    
    Xtest=1*(Xtest(ytestfilter, :)>0);
    ytest=ytest(ytestfilter, 1:filter);
    
end
    function [] = matlabgenereal()
        y = temp.gnd;
        
        X = temp.fea;
        if ~isfield(temp, 'testIdx')
            rand('seed', 0)
            C = cvpartition(y, 'kfold',2);
            temp.testIdx = C.test(1);
            temp.trainIdx = C.training(1);
        end
        y= to1ofk(y);
        Xtest = X(temp.testIdx,:);
        ytest = y(temp.testIdx,:);
        Xtrain = X(temp.trainIdx,:);
        ytrain = y(temp.trainIdx,:);
        Xu = Xtest;
    end

    function [] = getLIBSVMstd(trainpath, testpath, binary)
        if ~exist('binary', 'var')
            binary=1;
        end
        if ~exist('libsvmread', 'file')
            [Xtrain, ytrainind] = libsvmReadSparse(trainpath,binary);
            [Xtest, ytestind] = libsvmReadSparse(testpath,binary);
            ytrain=to1ofk(ytrainind);
            ytest=to1ofk(ytestind);
        else
            [ytrainind, Xtrain] = libsvmread(trainpath);
            [ytestind, Xtest] = libsvmread(testpath);
            if binary
                Xtrain = 1.0*(Xtrain>0);
                Xtest = 1.0*(Xtest>0);
            end
            ytrain=to1ofk(ytrainind+1);
            ytest=to1ofk(ytestind+1);
        end
        [Xtrain,Xtest] = makeEqual(Xtrain, Xtest);
        
        Xu = Xtest;
    end

    function [] = breakIntoParts()
        X = [Xtrain; Xtest];
        y = [ytrain; ytest];
        N = size(X,1);
        rand('seed', 0);
        randind = randperm(N);
        X = X(randind,:);
        y = y(randind,:);
        part = floor(N/3);
        Xtrain = X(1:part,:);
        Xtest = X(part+1:2*part,:);
        Xu = X(2*part+1:end,:);
        
        ytrain = y(1:part,:);
        ytest = y(part+1:2*part,:);
    end

end

function [A, B] = makeEqual(A,B)
    a = size(A,2); b = size(B,2);
    if a>b
        B = [B, zeros(size(B,1), a-b) ];
    elseif b>a
        A = [A, zeros(size(A,1), b-a) ];
    end
    B = [ones(size(B,1),1), B];
    A = [ones(size(A,1),1), A];

end

