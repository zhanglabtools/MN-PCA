function S = estimate_cov(X, iA, mode)
% Estimate the covariance by maximum likelihood estimation (`mle') or 
% well-conditioned estimation (`wce`).
% S = ESTIMATE_COV(X, iA, mode)
% Reference: Ledoit O, Wolf M. A well-conditioned estimator for 
%            large-dimensional covariance matrices[J]. 
%            Journal of multivariate analysis, 2004, 88(2): 365-411.

[n, p] = size(X);
Sn = X' * iA * X / n;
if strcmp('mle', mode)
    S = Sn;
elseif strcmp('wce', mode)
    m = trace(Sn) / p;
    d = sum(sum((Sn - m * eye(p)).^2));
    temp = iA * X;
%     for i = 1:n
%         b = b + sum(sum((X(i, :)' * temp(i, :) - Sn).^2)) / p;
%     end
    X2 = X.^2;
    temp2 = temp.^2;
    Sn2 = Sn.^2;
    b = sum(sum(X2.' * temp2 / n^2 - 2 * (X.' * temp ) .* Sn / n^2 + Sn2 / n));
    b = min(b, d);
    a = d - b;
    S = b / d * m * eye(p) + a / d * Sn;
elseif strcmp('corr', mode)
    u = sqrt(diag(Sn));
    S = diag(1 ./ u) * Sn * diag(1 ./ u);
else
    error('mode show be either `mle` or `wce`/n');
end
end

