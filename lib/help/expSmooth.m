function [S, issemipos] = expSmooth(D, sigma)
%Exponetial smooth function. D is the minkovski distance.
    eps = 1e-7;
    S = exp(-D.^2 / sigma);
    eigvals = eig(S);
    issemipos = all(eigvals>=-eps);
end

