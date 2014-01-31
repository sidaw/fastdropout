function [w, finalObj] = minFuncAdagrad(funObj,  W, X, y, options)
    eta = options.eta;
    w = W;
    numdata = size(X,1);
    G = 1e-10*ones(size(W));
    fprintf('Batchsize:%d\tMaxIter:%d\tNumdata:%d\n', ...
        options.BatchSize, options.MaxIter, numdata)
    for t = 1:options.MaxIter
        batchobj = 0;
        perm = randperm(length(y));
        X = X(perm, :);
        y = y(perm);
        for b = 1:ceil(size(X,1)/options.BatchSize)
            select = (b-1)* options.BatchSize+1:min(b* options.BatchSize, numdata);
            [finalObj, g] = funObj(w,X(select, :), y(select,:));
            g = g / length(select);
	        G = G + g.^2;
            finalObj = finalObj / length(select);
            batchobj = batchobj + finalObj;

            w = w - eta*g./sqrt(G);
        end
        fprintf('%d\t%f\t%f\n', t, batchobj, norm(g))
    end
end
