function [A, iA, B, iB] = init_gemini(Y, rank, lam1, lam2, eps)
%Estimate the sturcture of A and B with GEMINI
% [A, iA, B, iB, Out] = INIT_GEMINI(Y, rank, lam1, lam2)
    if ~exist('eps', 'var')
        eps = 0;
    end
    [U, S, W] =svds(Y, rank);
    X = U * S;
    R = Y - X * W';
    [n, p] = size(R);
    if n >= p
        f = p;
        m = n;
    else
        f = n;
        m = p;
        R = R';
    end
    s = corr(R);
    covB = R' * R / m;
    covB = covB + eps * eye(size(covB, 1));
    w2 = diag(covB);
    % step1
    [Brho, iBrho] = QUIC('default', s, lam2); 
    w2 = sqrt(sum(R.^2, 1));
    iB = diag(1./w2) * iBrho * diag(1./w2);
    B = diag(w2) * Brho *  diag(w2);
    % step2
    covA = R * iB * R' / f;
    covA = covA + eps * eye(size(covA, 1));
    w1 = sqrt(diag(covA));
    s = diag(1 ./ w1) * covA * diag(1 ./ w1);
    [Arho, iArho] = QUIC('default', s, lam1);  
    A = diag(w1) * Arho * diag(w1);
    iA = diag(1./w1) * iArho * diag(1./w1);
    covB = R' * iA * R / m;
    covB = covB + eps * eye(size(covB, 1));
    w2 = sqrt(diag(covB));
    s = diag(1 ./ w2) * covB * diag(1 ./ w2);
   [Brho, iBrho] = QUIC('default', s, lam2);
   B = diag(w2) * Brho * diag(w2);
   iB = diag(1./w2) * iBrho * diag(1./w2);
   if n < p
       temp = B;
       B = A;
       A = temp;
       temp = iB;
       iB = iA;
       iA = temp;
   end
    Out.avgdegree1 = (nnz(iA) - n) / n / 2;
    Out.avgdegree2 = (nnz(iB) - p) / p / 2;
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

