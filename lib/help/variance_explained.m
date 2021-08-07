function prop = variance_explained(Y, Q, R, U, V)
% Propotion of varianc explained in Q-R norm.
Pu = U / (U' * Q * U) * U';
Pv = V / (V' * R * V) * V';
Yk = Pu * Q * Y * R * Pv;
prop = trace(Q * Yk * R * Yk') / trace(Q * Y * R * Y');
end

