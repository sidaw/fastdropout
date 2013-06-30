function scalar = oneofktoscalar(y1ofk)
[~, scalar] = max(y1ofk, [], 2);