function [R, S1, S2] = cal_svd_residual(X, dim)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if dim > 0
    [U, D, V] = svds(X, dim);
else
    U = 0;
    D = 0;
    V = 0;
end
R = X - U * D * V';
S1 = R * R' / size(R, 2);
S2 = R' * R / size(R, 1);
end

