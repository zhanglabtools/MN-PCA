function [U, D, V] = gmd_method(Y, iA, iB, k, tol)
%Use power method to update U and V.
% [U, D, V] = gmd_method(Y, iA, iB, k, tol)
if ~exist('tol', 'var')
    tol = 1e-3;
end
[n, p] = size(Y);
d = zeros(k, 1);
U = zeros(n, k);
V = zeros(p, k);
Yk = Y;
for j = 1:k
    [uj, dj, vj] = power_method(Yk, iA, iB, tol);
    d(j) = dj;
    U(:, j) = uj;
    V(:, j) = vj;
    Yk = Yk - uj * dj * vj';
end
D = diag(d);
end

function val = Q_norm(X, Q)
val = trace(X' * Q * X);
val = sqrt(val);
end

function [uk, dk, vk] = power_method(Yk, Q, R, tol)
    [n, p] = size(Yk);
    max_iter = 100;
    uk = randn(n, 1);
    vk = randn(p, 1);
    YkQ = Yk' * Q;
    YkR = Yk * R;
    for n_iter = 1:max_iter
        pre_uk = uk;
        % fprintf('GPCA: niter=%d\n', n_iter);
        uk = YkR * vk /  Q_norm(YkR * vk, Q);
        if max(abs(pre_uk - uk)) < tol
            break;
        end
        vk = YkQ * uk / Q_norm(YkQ * uk, R);
    end
    dk = uk' * Q * YkR * vk;
end