function [iA, iB, info] = init_knn(X, k1, k2)
%INIT_KNN find k-nearest neighbor to initialize
% [iA, iB, info] = INIT_KNN(X, k1, k2)
[n, p] = size(X);
idx = knnsearch(X, X, 'K', k1);
iA = zeros(n, n);
for i = 1:n
    iA(i, idx(i, :)) = 1;
end
iA = (iA + iA' ) / 2;
idx = knnsearch(X', X', 'K', k2);
iB = zeros(p, p);
for i = 1:p
    iB(i, idx(i, :)) = 1;
end
iB = (iB + iB' ) / 2;
info.d1 = (nnz(iA) - n) / n / 2;
info.d2 = (nnz(iB) - p) / p / 2;
fprintf('Initialized avgdgree(iA) = %.2f avgdgree(iB) = %.2f \n', ...
             info.d1, info.d2);
end

