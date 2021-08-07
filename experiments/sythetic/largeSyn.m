clear;
addpath(genpath('./lib'));
addpath('MN-PCA-w2/');
addpath('MN-PCA-MRL/');
populations = [300, 300, 400];
n = 1000;
dim = 2;
sigmas = [0, 0, 0];
rc = 1 / 196;
spa = 1e-3;
p_v = [2000, 3000, 4000, 5000, 6000];
%% experiment setting
REP = 5;
n_iter_v = [300, 500, 500, 500, 500];
lam1_v = [.5, .5, .8, .8, .8];
lam2_v = [.5, .5, .8, .9, 1.0];
lamr_v = [0.05, 0.05, 0.05, 0.05, 0.05];

%% run experiments
genStr = genStrFun(REP, 'frob', 'psnr', 'rmse', 'nmi', 'prec1','prec2',...
                   'tpr1', 'tpr2', 'tnr1', 'tnr2');
for idx_p =  3 % 1:length(p_v)
fprintf('Size = %d\n', p_v(idx_p));
opts_td = struct('scale', 1, 'alpha_A', 1e-2, 'alpha_B', spa, 'rc_A', rc,...
              'rc_B', rc);
p = p_v(idx_p);
prop = 0.1;
centroids = zeros(3, p);
l = ceil(p * prop);
centroids(1, 1:l) = 1;
centroids(2, end-l:end) = -1;
centroids(3, end-l:end) = 1;
centroids(3, 1:l) = 1;
run_time_w2r = zeros(REP, 1);
run_time_mnff = zeros(REP, 1);
pca_res = genStr();
w2r_res = genStr();
mnff_res = genStr();
lam1 = lam1_v(idx_p);
lam2 = lam2_v(idx_p);
lamr = lamr_v(idx_p);
n_iter = n_iter_v(idx_p);
for idx = 1:REP
rng(idx)
td = toy_data2(populations, centroids,  0, opts_td);
true = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, 'nclust', 3, 'labels', td.labels);
% PCA
[pca_pred.score, pca_map] = pca(td.Y);
pca_pred = struct('signal', pca_pred.score * pca_map.M' + repmat(pca_map.mean, n, 1), 'score', pca_pred.score);
pca_res.evaluation(idx, 1:4) = helpEval(pca_res.metrics_names(1:4), true, pca_pred);
%% mnff
opts = struct('tol', 1e-3, 'est_cov', 'mle', 'lam1', lam1, 'lam2', lam2, 'r1', 1e-2, 'r2', 1e-2);
tic;
[mnff_X, mnff_W, mnff_iA, mnff_iB, ~] = MnPCA(td.Y, 2, opts);
[U, D, V] = gmd_method(td.Y, mnff_iA, mnff_iB, 2, 1e-4);
mnff_score = U * D;
run_time_mnff(idx) = toc;
%[mnff_score, ~] = pca(mnff_X * mnff_W');
mnff_pred = struct('signal', mnff_X * mnff_W', 'score', mnff_score); 
mnff_res.evaluation(idx, 1:4) = helpEval(mnff_res.metrics_names(1:4), true, mnff_pred);
% MN-PCA-w2
% [Y_, iA, iB] = MnPCAw2_wrapper(td.Y, dim, 1, 1, 1, true.signal, 200);
% w2_score = pca(Y_, 2);
% w2_pred = struct('signal', Y_, 'score', w2_score);
% w2_res.evaluation(idx, 1:4) = helpEval(w2_res.metrics_names(1:4), true, w2_pred);
% MN-PCA-w2r
tic;
[Y_r, iAr, iBr] = MnPCAq1_wrapper(td.Y, dim, lamr, lamr, 1, true.signal, n_iter);
run_time_w2r(idx) = toc;
w2r_score = pca(Y_r, 2);
w2r_pred = struct('signal', Y_r, 'score', w2r_score);
w2r_res.evaluation(idx, 1:4) = helpEval(w2r_res.metrics_names(1:4), true, w2r_pred);
fprintf('idx = %d\n', idx)
fprintf('   \tPCA\tMN-PCA\tMN-PCA-w2r\n');
fprintf('NMI\t%4.f\t%.4f\t%4.f\n', pca_res.evaluation(idx, 4),  mnff_res.evaluation(idx, 4), w2r_res.evaluation(idx, 4));
fprintf('run_time: %.4f\n', run_time_w2r(idx));
pause(1);
end
save(sprintf('../output_clean/largeSyn/vary_p=%dc=%d.mat', p, int32(1/ rc)),   ...
               'pca_res', 'w2r_res', 'mnff_res',... 
               'run_time_w2r', 'run_time_mnff', ...
               'n_iter', 'lam1', 'lam2', 'lamr');
end