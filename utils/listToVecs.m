function [X, y] = listToVecs(X, labels, wordsbi)

params.dictsize = length(wordsbi);
vecstrain = sparse(params.dictsize, length(labels));

for i = 1:length(labels)
    if mod(i,1000)==0
        disp(i)
    end
    x = X{i};
    if isempty(x); continue; end
    x=x(x>0);
    updates = unique(x);
    vecstrain(updates,i) = 1; %rawvecs(updates);
end


nInstances = length(labels);
nVars = params.dictsize;

X = [ones(nInstances,1) vecstrain'];
y = sign(2*labels'-1);

end