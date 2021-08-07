function bic = BIC(Y, iA, iB, X, W)
%Calculate BIC.
% bic = BIC(Y, iA, iB, X, W)
[n, p] = size(Y);
R = Y - X * W';
bic = trace(iA * R * iB * R') /(n * p);
bic = bic -logdet(iA) / p - logdet(iB) / n;
bic = bic + (nnz(iA) - n) / 2 * log(p) / (n * p);
bic = bic + (nnz(iB) - p) / 2 * log(n) / (n * p);
end

function x = logdet(A)
L = chol(sparse(A), 'lower');
x =  sum(log(diag(L))) * 2;
end