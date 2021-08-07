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
% rc_v = 1 ./[8, 16, 32, 64, 96, 128, 160, 192, 224];
rc = 1 / 192;
REP = 10;
spa = 0.01;
lam_v = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
% Construct function to gen stru to save metrics.
genStr = genStrFun(REP, 'frob', 'psnr', 'rmse', 'nmi', 'prec1','prec2',...
                   'tpr1', 'tpr2', 'tnr1', 'tnr2');
for i = 7:length(lam_v)
    opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc,...
              'rc_B', rc);
    w2_res = genStr();
    for idx = 1:REP
        % generate data
        rng(idx)
        td = toy_data2(populations, centroids, sig, opts_td);
        true_data = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, ...
            'nclust', 3, 'labels', td.labels);
        [Y, iA, iB] = MnPCAq1_wrapper(td.Y, 2, lam_v(i), lam_v(i), 1, td.Y - td.E, 500);
        w2_score = pca(Y);
        w2_pred = struct('signal', Y, 'score', w2_score, 'iA', iA, 'iB', iB);
        w2_res.evaluation(idx, :) = helpEval(w2_res.metrics_names, true_data, w2_pred);
    end
    fprintf('save file to ./output/small_syn/w2_lam=%.2f.mat\n', lam_v(i));
    save(sprintf('./output/small_syn/w2_lam=%.2f.mat', lam_v(i)), 'w2_res');
end
%% Print table
clear;
addpath(genpath('./lib'));
rc = 1 / 192;
lam_v = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
psnr_vary_lam_mean = zeros(length(lam_v), 1);
psnr_vary_lam_std = zeros(length(lam_v), 1);
nmi_vary_lam_mean = zeros(length(lam_v), 1);
nmi_vary_lam_std = zeros(length(lam_v), 1);
tpr1_vary_lam = zeros(length(lam_v), 1);
tpr2_vary_lam = zeros(length(lam_v), 1);
prec1_vary_lam = zeros(length(lam_v), 1);
prec2_vary_lam = zeros(length(lam_v), 1);


for lam_idx = 1:length(lam_v)
    res_lam = load(sprintf('./output/small_syn/w2_lam=%.2f.mat', lam_v(lam_idx)));
    res_lam = res_lam.w2_res;
    psnr_vary_lam_mean(lam_idx) = mean(res_lam.evaluation(:, 2));
    psnr_vary_lam_std(lam_idx) = std(res_lam.evaluation(:, 2));
    nmi_vary_lam_mean(lam_idx) = mean(res_lam.evaluation(:, 4));
    nmi_vary_lam_std(lam_idx) = std(res_lam.evaluation(:, 4));
    tpr1_vary_lam(lam_idx) = mean(res_lam.evaluation(:, 7));
    tpr2_vary_lam(lam_idx) = mean(res_lam.evaluation(:, 8));
    prec1_vary_lam(lam_idx) = mean(res_lam.evaluation(:, 5));
    prec2_vary_lam(lam_idx) = mean(res_lam.evaluation(:, 6));
end
my_blue = [0 0.4470 0.7410];
my_orange = [0.8500 0.3250 0.0980];
offset = 0.05;
subplot(1, 2, 1)
errorbar(lam_v, psnr_vary_lam_mean, psnr_vary_lam_std, '--o',  'color', my_blue, ...
    'MarkerEdgeColor', my_blue,'MarkerFaceColor', my_blue)
xlabel('\lambda')
ylabel('PSNR')
xlim([-offset, 1 + offset])
subplot(1, 2, 2)
errorbar(lam_v, nmi_vary_lam_mean / 100, nmi_vary_lam_std / 100, '--o',  'color', my_orange, ...
    'MarkerEdgeColor', my_orange,'MarkerFaceColor', my_orange)
xlabel('\lambda')
ylabel('NMI')
xlim([-offset, 1 + offset])
save('./output/small_syn/w2_vary_lam.mat', ...
    'psnr_vary_lam_mean', 'psnr_vary_lam_std', ...
    'nmi_vary_lam_mean', 'nmi_vary_lam_std', ...
    'tpr1_vary_lam', 'tpr2_vary_lam', ...
    'prec1_vary_lam', 'prec2_vary_lam');