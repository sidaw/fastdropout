function y1ofk = to1ofk(y, k)
y=y(:);
y= y-min(y)+1;
n = length(y);

if nargin==1
k = max(y);
end

y1ofk = zeros(n, k);
y1ofk(sub2ind([n,k], 1:n, y'))=1;