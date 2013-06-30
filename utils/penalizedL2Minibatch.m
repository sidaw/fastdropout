function [nll,g,H] = penalizedL2Minibatch(w,Xtrain,ytrain,gradFunc,lambda,varargin)
% Adds Kernel L2-penalization to a loss function
% (you can use this instead of always adding it to the loss function code)

if isscalar(lambda)
    lambda = lambda*ones(size(w));
end

if nargout <= 1
    [nll] = gradFunc(w,Xtrain,ytrain,varargin{:});
elseif nargout == 2
    [nll,g] = gradFunc(w,Xtrain,ytrain,varargin{:});
else
    [nll,g,H] = gradFunc(w,Xtrain,ytrain,varargin{:});
end

nll = nll+sum(lambda.*(w.^2));

if nargout > 1
    g = g + 2*lambda.*w;
end

if nargout > 2
    H = H + diag(2*lambda);
end