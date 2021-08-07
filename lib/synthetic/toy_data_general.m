function td = toy_data_general(populations, centroids, sig, opts, noise_type)

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
td.E = general_mnormal(A, B, noise_type) * opts.scale;
td.Y = td.E + td.X;
end

%% local funtion
function y  = laprnd(m, n, mu, sigma)
%LAPRND generate i.i.d. laplacian random number drawn from laplacian distribution
%   with mean mu and standard deviation sigma. 
%   mu      : mean
%   sigma   : standard deviation
%   [m, n]  : the dimension of y.
%   Default mu = 0, sigma = 1. 
%   For more information, refer to
%   http://en.wikipedia.org./wiki/Laplace_distribution
%   Author  : Elvis Chen (bee33@sjtu.edu.cn)
%   Date    : 01/19/07
%Check inputs
    if nargin < 2
        error('At least two inputs are required');
    end
    if nargin == 2
        mu = 0; sigma = 1;
    end
    if nargin == 3
        sigma = 1;
    end
    % Generate Laplacian noise
    u = rand(m, n)-0.5;
    b = sigma / sqrt(2);
    y = mu - b * sign(u).* log(1- 2* abs(u));
end

function E = gaussian_mixture(m, n, props, scales)
    % generate gaussian mixture noise
    assert(length(props) == length(scales));
    mask = rand(m, n);
    E = 0;
    cum_props = [0, cumsum(props)];
    for ind = 1:length(props)
        E = E + ((mask <= cum_props(ind+1)) & (mask > cum_props(ind))) ...
            .* randn(m, n) * scales(ind); 
    end
end

function E = general_mnormal(A, B, noise_type)
% MNORMAL generate matrix normal data with MN(0, A, B)
    La = chol(A, 'lower');
    Lb = chol(B, 'lower');
    [m, ~] = size(A);
    [n, ~] = size(B);
    if strcmp(noise_type, 'Laplace')
        E = laprnd(m, n, 0, 1);
    elseif strcmp(noise_type, 'uniform')
       % U(-1, 1)
       E = (rand(m, n) - 0.5) * 2;
    elseif strcmp(noise_type, 'mixture1')
        % 30% norm(0.5, 1)  + 30% norm(0.2, 1) + 40% norm(0, 1);
        E = gaussian_mixture(m, n, [0.3, 0.3, 0.4], [sqrt(0.2), sqrt(0.5), 1]);
    elseif strcmp(noise_type, 'mixture2')
        % 40% norm(0, 0.5)  + 50% norm(0, 1) + 10% norm(0, 3)
        E = gaussian_mixture(m, n, [0.4, 0.5, 0.1], [sqrt(0.5), 1, sqrt(3)]);
    elseif strcmp(noise_type, 'mixture3')
        % 45% norm(0, 0.5)  + 45% norm(0, 1) + 10%
        mask = rand(m, n);
        E = (randn(m, n) * sqrt(0.5)) .* (mask < 0.45);
        E = E + (randn(m, n) * sqrt(0.5)) .* ((mask >= 0.45) & (mask < 0.9));
        E = E + ((rand(m, n) - 0.5) * 2) .* (mask >= 0.9) ;
    else
        error("Unknown noise type.")
    end
    E = La * E * Lb;
end

 