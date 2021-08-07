function [L, issemipos] = lapDist(D, window)
    eps = 1e-7;
    n = size(D, 1);
    L = zeros(n, n);
    nnzL = D + eye(size(D)) * window < window;
    idx = find(nnzL);
    L(idx) = - 1./ D(idx);
    L(1:n+1:end) = -sum(L, 1);
    eigvals = eig(L);
    L = L / max(eigvals);
    issemipos = all(eigvals >= -eps);
    if any(isnan(eigvals))
        issemipos = 0;
    end
end