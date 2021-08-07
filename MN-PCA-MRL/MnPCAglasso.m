function [X, W, iA, iB, Out] = MnPCAglasso(Y, dim, opts)
% MnPCA Matrix-normal PCA 
%   [X, W, iA, iB, Out] = MNPCA(Y, dim, opts.lam1, opts.lam2, opts)
%   update inverse matrix with Glasso
% Input
%   Y: n * p input data matrix
%   dim: reduction dimension
%   opts: options struct contains following fileds
%           .lam1: l1 regularization on iA
%           .lam2: l1 regularization on iB 
%           .tol: tolerance
%           .max_iter: maximum number of iteration
%           .verbose: binary
%           .est_cov: method to estimate the correlation matrix. 'mle' or
%           'wce'
%           .g1, r1: l1 and l2 regularizaitons on X, default 0
%           .g1, r2: l1 and l2 regularizaitons on W, default 0
%
%% process options
if ~isfield(opts, 'tol')
    opts.tol = 1e-4;
end
if ~isfield(opts, 'max_iter')
    opts.max_iter = 20;
end
if ~isfield(opts, 'verbose')
    opts.verbose = 0; 
end
if ~isfield(opts, 'est_cov')
    opts.est_cov = 'mle';
end
% if ~isfield(opts, 'dmin1')
%     opts.dmin1 = 1e-3;
% end
% if ~isfield(opts, 'dmax1')
%     opts.dmax1 = 1e3;
% end
% if ~isfield(opts, 'dmin2')
%     opts.dmin2 = 1e-3;
% end
% if ~isfield(opts, 'dmax2')
%     opts.dmax2 = 1e3;
% end
abs_tol = 1e-4;
reg_names = {'g1', 'r1', 'g2', 'r2'};
for i = 1:length(reg_names)
    if ~isfield(opts, reg_names{i})
        opts.(reg_names{i}) = 0;
    elseif opts.(reg_names{i}) < 0
        error('Illegal regularizaiton term. %s must be positive.\n', ...
             opts.(reg_names{i}));
    end
end
[n, p] = size(Y);    
Out.objvals = zeros(opts.max_iter + 1, 1);
% initilized with SVD.
[Uk, sk, W] = svds(Y, dim);
X = Uk * sk;
R = Y - X * W';
iA = eye(n);
iB = eye(p); 
val = obj_val(Y, X, W, iA, iB, opts);
Out.objvals(1) = val;
if opts.verbose
    fprintf('Initilized objval=%.4f\n', Out.objvals(1)) ;
end

for i = 2:opts.max_iter
    % update iA
        if opts.lam1 > 0
            S1 = estimate_cov(R', iB, opts.est_cov);
            if isfield(opts, 'mask1')
                S1 = mask_cov(S1, opts.mask1);
            end
% [X W opt time iter dGap] = QUIC("default", S, L, tol, msg, ...
%                                 maxIter, X0, W0)            
            [~, iA, ~, ~, ~] = myGLasso(S1, opts.lam1);
%             [iA, ~] = QUIC('default', S1, opts.lam1, tol, 0, ...
%             QUIC_MAXITER, iA, inv(iA));
    %         % normalize iA
    %         u = sqrt(diag(iA));
    %         iA = diag(1 ./ u) * iA * diag(1 ./ u);
            if opts.verbose
                val = obj_val(Y, X, W, iA, iB, opts);
                fprintf('Update iA: %.4f\n', val);
            end
            if isfield(opts, 'dmin1') && sfield(opts, 'dmax1')
                iA = normalize_precision(iA, opts.dmin1, opts.dmax1);
                if opts.verbose
                    val = obj_val(Y, X, W, iA, iB, opts);
                    fprintf('Nomalize iA: %.4f\n', val);
                end
            end
            iA = sparse(iA);
        end
        % update iB
        if opts.lam2 > 0
            S2 = estimate_cov(R, iA, opts.est_cov);
            if isfield(opts, 'mask2')
                S2 = mask_cov(S2, opts.mask2);
            end            
            % [iB, ~] = QUIC('default', S2, opts.lam2);
            [~, iB, ~, ~, ~] = myGLasso(S2, opts.lam2);
            % warm start
            % [iB, ~] = QUIC('default', S2, opts.lam2, 1e-4, 0, ...
            % QUIC_MAXITER, iB, inv(iB));
            if isfield(opts, 'dmin2') && isfield(opts, 'dmax2')
                iB = normalize_precision(iB, opts.dmin2, opts.dmax2);
                if opts.verbose
                    val = obj_val(Y, X, W, iA, iB, opts);
                    fprintf('Update iB: %.4f\n', val);
                end
            end
            iB = sparse(iB);
        end
    % update X 
    pre_val = partial_obj_val(Y, X, W, iA, iB, opts);
    for sub_iter = 1:20
        X = reg_ls(Y, X, W, iA, iB, opts.g1, opts.r1);
        val = partial_obj_val(Y, X, W, iA, iB, opts);
        if abs(pre_val - val) < 1e-4
            break
        end
        pre_val = val;
        if opts.verbose
            val = obj_val(Y, X, W, iA, iB, opts);
            fprintf('Update X: %.4f\n', val);
        end
        W = reg_ls(Y', W, X, iB, iA, opts.g2, opts.r2);
        if opts.verbose
            val = obj_val(Y, X, W, iA, iB, opts);
            fprintf('Update W: %.4f\n', val);
        end

    end
    R = Y - X * W';
    Out.objvals(i) = obj_val(Y, X, W, iA, iB, opts);
    fprintf('Iter %d: ObjVal=%.4f\n', i, Out.objvals(i));
    % stop criterion
    DeltaObj = abs(Out.objvals(i) - Out.objvals(i - 1)) / (abs(Out.objvals(i - 1))+1e-5);
    fprintf('DeltaObj: %% %.2f\n', DeltaObj * 100);
    if Out.objvals(i) > Out.objvals(i-1)
        break; % bad iteration
    end
    if DeltaObj < opts.tol || abs(Out.objvals(i) - Out.objvals(i - 1)) < abs_tol
        break;
    end
end
Out.objvals = Out.objvals(1:i);
end
%% local function
function val = off_diag_sum(A)
% calculate off diagonal summation.
val = sum(sum(abs(A - diag(diag(A)))));
end

function x = logdet(A)
L = chol(sparse(A), 'lower');
x =  sum(log(diag(L))) * 2;
end

function val = obj_val(Y, X, W, iA, iB, opts)
% L = log|A| + log|B| * p / n + tr(iB * R' * iA * R) / n + opts.lam1 * |iA|_1 +
% opts.lam2 * |iB|_2 + g1 * |X|_1 + r1 * |X|_2 +  g2 * |W|_1 + r2 * |W|_2
[n, p] = size(Y);
R = Y - X * W';
val = -logdet(iA) / n  - logdet(iB) / p; 
val = val + trace(iB * R' * iA * R) / (n*p);
if isscalar(opts.lam1)
    val = val + opts.lam1 * off_diag_sum(iA) / p;
else
    val = val + sum(sum(opts.lam1 .* iA));
end
if isscalar(opts.lam2)
    val = val + opts.lam2 * off_diag_sum(iB) / n;
else
    val = val + sum(sum(opts.lam2 .* iB));
end

val = val + (opts.r1 * sum(sum(X.^2)) + opts.g1 * sum(sum(abs(X)))) / (n * p);
val = val + (opts.r2 * sum(sum(W.^2)) + opts.g2 * sum(sum(abs(W)))) / (n * p);
end

function X = normalize_precision(X, dmin, dmax)
[Q, L] = eig(X);
es = diag(L);
es = min(max(dmin, es), dmax);
X = Q * diag(es) * Q';
X(abs(X) < 1e-5) = 0;
end

function val = partial_obj_val(Y, X, W, iA, iB, opts)
[n, p] = size(Y);
R = Y - X * W';
val = trace(iB * R' * iA * R) / (n*p);
val = val + (opts.r1 * sum(sum(X.^2)) + opts.g1 * sum(sum(abs(X)))) / (n * p);
val = val + (opts.r2 * sum(sum(W.^2)) + opts.g2 * sum(sum(abs(W)))) / (n * p);
end

function S_ = mask_cov(S, mask)
    S_ = S .* mask;
end