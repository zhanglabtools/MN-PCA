function lam = lam_matrix(L, high, low)
% ragularizaiton matrix lam
lam = ones(size(L)) * high;
lam(L > 0) = low;
lam = lam - diag(diag(lam));
end

