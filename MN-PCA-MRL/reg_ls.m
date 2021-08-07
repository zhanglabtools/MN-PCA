function X = reg_ls(Y, X, W, iA, iB, g1, r1)
%Regularized least square   
tol = 1e-3;
if g1 == 0
    X = Y * iB *W / (W' * iB * W + r1 * eye(size(W, 2)));
else
    error('Not implemented.\n');
end
end

