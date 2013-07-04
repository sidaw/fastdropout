function [X, y, comments] = libsvmReadSparse(fpath, binary)
% Relatively fast matlab to read sparse data in libsvm format
% Adapted to sparse data by sidaw
if nargout<2
    binary = 0;
end

raw = cellfun(@strtrim, getText(fpath), 'UniformOutput', false);
iscomment = cellfun(@(s)s(1) == '#', raw);
comments = raw(iscomment);
data  = raw(~iscomment);
ragged = cellfun(@(c)str2num(strrep(c, ':', ' ')), data, 'UniformOutput', false);

n = numel(ragged);
d = max(cellfun(@(c)c(end-1), ragged, 'ErrorHandler', @(varargin)0));
Xinds = zeros(n*100, 3);
y = zeros(n, 1);
begin = 1;
for i=1:n
    row = ragged{i};
    y(i) = row(1);
    
    inds = row(2:2:end);
    finish = begin+length(inds)-1;
    if binary
        values = row(3:2:end) > 0;
    else
        values = row(3:2:end);
    end
    Xinds(begin:finish,1) = i;
    Xinds(begin:finish,2) = inds;
    Xinds(begin:finish,3) = values;
    begin=begin+length(inds);
    
end
Xinds = Xinds(1:finish,:);
X = spconvert(Xinds);
end
