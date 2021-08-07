%% add path
% In current setting. Report the results of 
% add path
clear
addpath(genpath('./lib'));
addpath('MN-PCA-w2/');
addpath('MN-PCA-MRL/');
%% Experiments Setting
populations = [100, 100, 100];
n = sum(populations);
p = 200;
c = length(populations);
centroids = zeros(c, p);
l = 20;
rng('default')
rng(0)
centroids(1, 1:l) = 1;
centroids(1, end-l:end) = 1;
centroids(2, 1:l) = -1;
centroids(2, end-l+1:end) = -1;
centroids(3, 1:l) = 1;
centroids(3, end-l+1:end) = -1;
dim = 2;
scale = 1;
sig = 0;
% centroids = centroids * 1.2;
% centroids(3, 2 * l:end-l) = 1;
% rc_v = 1 ./[1, 4, 8, 12, 16];
rc_v = 1 ./[8, 16, 32, 64, 96, 128, 160, 192, 224];
REP = 10;
spa = 0.01;
run_methods = {'PCA', 'MN-PCA', 'W2', 'W2r'};
%% Construct function to gen stru to save metrics.
genStr = genStrFun(REP, 'frob', 'psnr', 'rmse', 'nmi', 'prec1','prec2',...
                   'tpr1', 'tpr2', 'tnr1', 'tnr2');
%% Experiments               
for i = 1:length(rc_v)
opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc_v(i),...
              'rc_B', rc_v(i));
td = toy_data2(populations, centroids, sig, opts_td);
[cand_lam1, cand_lam2] = cand_lam(td.Y, 2, .05, 3, 20);
[lam1, lam2, score1, score2] = choose_lam(td.Y, 2, cand_lam1, cand_lam2);
% struct to save results
pca_res = genStr();
mnff_res = genStr();
w2_res = genStr();
w2r_res = genStr();
for idx = 1:REP
%% generate data
rng(idx)
td = toy_data2(populations, centroids, sig, opts_td);
true = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, 'nclust', 3, 'labels', td.labels);
%% run methods
run_results = {};
if ismember('PCA', run_methods)
    [pca_pred.score, pca_map] = pca(td.Y);
    pca_pred = struct('signal', pca_pred.score * pca_map.M' + repmat(pca_map.mean, n, 1), 'score', pca_pred.score);
     
    run_results{end+1} = pca_res;
end
if ismember('MN-PCA', run_methods)
    opts = struct('tol', 1e-3, 'est_cov', 'mle', 'lam1', lam1, 'lam2', lam2, 'r1', 1e-2, 'r2', 1e-2);
    [mnff_X, mnff_W, mnff_iA, mnff_iB, ~] = MnPCA(td.Y, 2, opts);
    [U, D, V] = gmd_method(td.Y, mnff_iA, mnff_iB, 2, 1e-4);
    mnff_score = U * D;
    mnff_pred = struct('signal', mnff_X * mnff_W', 'score', mnff_score, 'iA', mnff_iA, 'iB', mnff_iB);
    mnff_res.evaluation(idx, :) = helpEval(mnff_res.metrics_names, true, mnff_pred);
    run_results{end+1} = mnff_res;
end
if ismember('W2', run_methods)
    [Y_, iA, iB] = MnPCAw2_wrapper(td.Y, 2, 1, 1, 1, td.Y - td.E, 500);
    w2_score = pca(Y_);
    w2_pred = struct('signal', Y_, 'score', w2_score, 'iA', iA, 'iB', iB); 
    w2_res.evaluation(idx, :) = helpEval(w2_res.metrics_names, true, w2_pred);
    run_results{end+1} = w2_res;
end
if ismember('W2r', run_methods)
    [Y_r, iAr, iBr] = MnPCAq1_wrapper(td.Y, 2, 0.05, 0.05, 1, td.Y - td.E, 500);
    w2r_score = pca(Y_r);
    w2r_pred = struct('signal', Y_r, 'score', w2r_score, 'iA', iAr, 'iB', iBr); 
    w2r_res.evaluation(idx, :) = helpEval(w2r_res.metrics_names, true, w2r_pred);
    run_results{end+1} = w2r_res;
end
%%  print reulsts

fprintf('rc = %d\n\t', 1 / rc_v(i))
for j = 1:length(run_methods)
    fprintf('%s\t', run_methods{j})
end
fprintf('\nNMI\t')
for j = 1:length(run_methods)
    res = run_results{j};
    fprintf('%.2f\t', res.evaluation(idx, 4));
end
fprintf('\ntpr1\t')
for j = 1:length(run_methods)
    res = run_results{j};
    fprintf('%.2f\t', res.evaluation(idx, 7));
end
fprintf('\nprec1\t')
for j = 1:length(run_methods)
    res = run_results{j};
    fprintf('%.2f\t', res.evaluation(idx, 5));
end
fprintf('\ntnr1\t')
for j = 1:length(run_methods)
    res = run_results{j};
    fprintf('%.2f\t', res.evaluation(idx, 9));
end
fprintf('\n')
pause(1)   
end
save(sprintf('../output_clean/smallSyn/syn_results_c=%d.mat', 1/ rc_v(i)), 'pca_res',  'mnff_res', 'w2_res', 'w2r_res');% , 'other_res');
end
