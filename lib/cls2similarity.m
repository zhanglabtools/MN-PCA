function S = cls2similarity(cls)
%CLS2CONSEN convert multiple output of clustering to similarity matrix
%   similarity = CLS2SIMILARITY
%   cls [n, p] where n is the number of samples, p is the number of clustering
%   output
[n, p] = size(cls);
S = zeros(n, n);
for i = 1:n
    for j = (i+1):n
        S(i, j) = sum(cls(i, :) == cls(j, :));
    end
end
S = S / p;
S = S + S' + eye(n);
end

