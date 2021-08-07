function td = toy_data(populations, centroids, sigmas, n, opts)
%TOY_DATA generate toy data for simulation.
%   populations: array of length c.
%   centriods: d * c matrix denote centriods.
[d, c] = size(centroids); % num
assert(c == length(populations));
m = sum(populations(:)); 
td.labels = repeat(1:c, populations);
% process options 
if isfield(opts, 'seed')
    rng(opts.seed);
elseif ~isfield(opts, 'alpha_A')
    opts.alpha_A = 0.02;
elseif ~isfield(opts, 'alpha_B')
    opts.alpha_B = 0.02;
elseif ~isfield(opts, 'scale')
    opts.scale = 1.0;
elseif ~isfield(opts, 'rc_A')
    opts.rc_A = 1.0;
elseif ~isfield(opts, 'rc_B')
    opts.rc_B = 1.0;
end

X = zeros(m, n);
st = 1;
for idx = 1:c
    part_X = normrnd(0, sigmas(idx), populations(idx), d);
    part_X = bsxfun(@plus, part_X, centroids(:, idx)');
    X(st:st + populations(idx) - 1, 1:d) = part_X;
    st = st + populations(idx);
end
perm_idx = randperm(m);
td.labels = td.labels(perm_idx);
td.X = X(perm_idx, :);
% generate matrix normal noise 
td.iA = sprandsym(m, opts.alpha_A, opts.rc_A, 1);
td.iB = sprandsym(n, opts.alpha_B, opts.rc_B, 1);
% diagonal to 1.
diA = sqrt(1 ./ diag(td.iA));
td.iA = diag(diA) * td.iA * diag(diA);
diB = sqrt(1 ./ diag(td.iB));
td.iB = diag(diB) * td.iB * diag(diB);
% add matrix normal noise
A = inv(full(td.iA));
B = inv(full(td.iB));
td.E = mnormal(A, B) * opts.scale;
td.Y = td.X + td.E;
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

function x = repeat(arr, reps)
x = zeros(sum(reps(:)), 1);
st = 1;
for idx = 1:length(reps)
    x(st:st + reps(idx) - 1) = arr(idx);
    st = st + reps(idx);
end
end