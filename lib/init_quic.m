function [A, iA, B, iB, Out] = init_quic(Y, rank, lam1, lam2, mode)
%Use QUIC to initialize mnmfsp
%   [A, iA, B, iB] = INIT_QUIC(Y, rank, lam)
    [U, S, W] =svds(Y, rank);
    X = U * S;
    R = Y - X * W';
    [m, n] = size(R);
    if ~exist('mode', 'var')
        mode = 'corr';
    end
    Sa = R * R' / n;
    Sb = R' * R / m;
    if strcmp(mode, 'cov');   
    elseif strcmp(mode, 'corr')
        w = diag(Sa);
        w = sqrt(diag(1./w));
        Sa = w * Sa * w;
        w = diag(Sb);
        w = sqrt(diag(1./w));
        Sb = w * Sb * w;
    else
        error('mode must be cov or corr');
    end
    [iA, A] = QUIC('default', Sa, lam1);
    [iB, B] = QUIC('default', Sb, lam2);
    Out.avgdegree1 = (nnz(iA) - m) / m / 2;
    Out.avgdegree2 = (nnz(iB) - n) / n / 2;
    Out.rc1 = 1 / cond(iA);
    Out.rc2 = 1 / cond(iB);
    fprintf('Initialized avgdgree(iA) = %.2f avgdgree(iB) = %.2f \n', ...
             Out.avgdegree1, Out.avgdegree2);
    fprintf('Reciprocal condition number rc(iA) = %.2f, rc(iB) = %.2f \n', ...
            Out.rc1, Out.rc2);
    if max(Out.avgdegree1, Out.avgdegree2) < 1e-3
        warning('Initilized iA or iB is too sparse. Try a smaller lam.');
    elseif max(Out.avgdegree1, Out.avgdegree2) > 10
        warning('Initialized iA or iB is too dense. Try a bigger lam.');
    end
end

