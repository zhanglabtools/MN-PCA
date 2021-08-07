function [A_, u] = diag_normalize(A)
%UNTITLED6 Summary of this function goes here
    u = sqrt(diag(A));
    A_ = diag(1 ./ u) * A * diag(1 ./ u);
end

