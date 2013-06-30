function varargout = stack2param(X, decodeInfo)
% i think this is from Richard Socher or the UFLDL tutorial
assert(length(decodeInfo)==nargout,'this should output as many variables as you gave to get X with param2stack!')

index=0;
for i=1:length(decodeInfo)
    if iscell(decodeInfo{i})
        for c = 1:length(decodeInfo{i})
            matSize = decodeInfo{i}{c};
            cellOut{c} = reshape(X(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
            index = index+(matSize(1))*matSize(2);
        end
        varargout{i}=cellOut;
    else
        matSize = decodeInfo{i};
        varargout{i} = reshape(X(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
        index = index+(matSize(1))*matSize(2);
    end
end
