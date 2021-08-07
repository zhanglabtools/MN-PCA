function td = toy_data2(populations, centroids, sig, opts)

%td = TOY_DATA2(populations, centroids, sigmas, n, opts)
%   td.Y: n * p data matrix
%   populations: array of length c.
%   centriods: d * c matrix denote centriods.
[c, p] = size(centroids); % num
assert(c == length(populations));
n = sum(populations(:)); 
% process options 
if isfield(opts, 'seed')
    rng(opts.seed);
end
if ~isfield(opts, 'alpha_A')
    opts.alpha_A = 0.02;
end
if ~isfield(opts, 'alpha_B')
    opts.alpha_B = 0.02;
end
if ~isfield(opts, 'scale')
    opts.scale = .1;
end
if ~isfield(opts, 'rc_A')
    opts.rc_A = 1.0;
end
if ~isfield(opts, 'rc_B')
    opts.rc_B = 1.0;
end
td.labels = repelem(1:c, populations);
X = centroids(td.labels, :);
perm_idx = randperm(n);
td.labels = td.labels(perm_idx);
td.X = X(perm_idx, :) + sig * randn(size(X));
% generate matrix normal noise 
td.iA = sprandsym(n, opts.alpha_A, opts.rc_A, 1);
td.iB = sprandsym(p, opts.alpha_B, opts.rc_B, 1);
% diagonal to 1.
diA = sqrt(1 ./ diag(td.iA));
td.iA = diag(diA) * td.iA * diag(diA);
diB = sqrt(1 ./ diag(td.iB));
td.iB = diag(diB) * td.iB * diag(diB);
% add matrix normal noise
A = inv(full(td.iA));
B = inv(full(td.iB));
td.E = mnormal(A, B) * opts.scale;
td.Y = td.E + td.X;
end

%% local funtion
function E = mnormal(A, B)
% MNORMAL generate matrix normal data with MN(0, A, B)
La = chol(A, 'lower');
Lb = chol(B, 'lower');
[m, ~] = size(A);
[n, ~] = size(B);
E = randn(m, n);
E = La * E * Lb;
end