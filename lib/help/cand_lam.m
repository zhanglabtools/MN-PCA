function [cand_lam1, cand_lam2] = cand_lam(Y, dim, ratio, maxrho, num)
% [cand_lam1, cand_lam2] = CAND_LAM(Y, dim)
if ~exist('num', 'var')
    num = 10;
end
if ~exist('ratio', 'var')
    ratio = .1;
end
if ~exist('maxrho', 'var')
    maxrho = inf;
end
min_rho = 0.05;
[~, S1, S2] = cal_svd_residual(Y, dim);
rho1 = S1 - eye(size(S1));
rho1 = max(rho1(:)) - min(rho1(:));
rho1 = min(rho1, maxrho);
cand_lam1 = logspace(log10(max(min_rho, rho1 * ratio)), log10(rho1), num);
rho2 = S2 - eye(size(S2));
rho2 = max(rho2(:)) - min(rho2(:));
rho2 = min(rho2, maxrho);
cand_lam2 = logspace(log10(max(min_rho, rho2 * ratio)), log10(rho2), num);
end

