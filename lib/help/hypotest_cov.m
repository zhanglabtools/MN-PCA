function [T, p] = hypotest_cov(S, n)
% Hypotheis test H0: S = I H1: S != I
% John's test
p = size(S, 1);
W = sum(sum((S - eye(p)).^2)) / p - p / n * (1 / p * trace(S))^2 + p / n;  
T = n * W - p - 1; % following chi-square distribution with degeree p(p+1) / 2
T = T * .5;
if T < 0
    p = normcdf(T) * 2;
else
    p = (1 - normcdf(T)) * 2;
end
end