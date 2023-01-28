# The basic usage of MN-PCA
Matrix normal PCA provides two algorithms to obtain low-rank representaion and the two-way noise structure.

First add folders to the working path.
``` matlab
addpath(genpath('./lib'));
addpath('MN-PCA-w2/');
addpath('MN-PCA-MRL/');
```
Generate the toy data and visualize it.
``` matlab
populations = [100, 100, 100];
n = sum(populations);
p = 200;
c = length(populations);
centroids = zeros(c, p);
l = 20;
rng('default')
centroids(1, 1:l) = 1;
centroids(1, end-l:end) = 1;
centroids(2, 1:l) = -1;
centroids(2, end-l+1:end) = -1;
centroids(3, 1:l) = 1;
centroids(3, end-l+1:end) = -1;
dim = 2;
scale = 1;
sig = 0;
rc_v = 1 ./[32, 192];
spa = 0.01;
sz = 6;
set_fig('units','inches','width', 7.0,'height', 3.75,'font','Times New Roman','fontsize', 8)
palettes = cbrewer('qual', 'Set1', 3); 
```

Run experiments on toy data.
``` Matlab 
%% Run experiments
for idx = 1:length(rc_v)
    rng(3210);
    % toy data setting
    opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc_v(idx),...
              'rc_B', rc_v(idx));
    td = toy_data2(populations, centroids, sig, opts_td);
    true = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, ...
        'nclust', 3, 'labels', td.labels);
    % PCA
    [pca_pred.score, pca_map] = pca(td.Y);
    pca_pred = struct('signal', pca_pred.score * pca_map.M' + repmat(pca_map.mean, n, 1), ...
        'score', pca_pred.score);
        subplot(2, 3, 3 * (idx - 1) + 1);
    plotScatter(pca_pred.score, true.labels, palettes, sz);
    %xlabel('PC1')
    ylabel('PC2')
    title('PCA', 'fontSize', 10, 'FontWeight', 'Normal')
    if idx == 2
        xlabel("PC1")
    end
    drawnow;
    %% MN-PCA-MRL
    % choose lambda for MN-PCA
    [cand_lam1, cand_lam2] = cand_lam(td.Y, 2, .05, 3, 20);
    [lam1, lam2, score1, score2] = choose_lam(td.Y, 2, cand_lam1, cand_lam2);
    opts = struct('tol', 1e-3, 'est_cov', 'mle', 'lam1', lam1, 'lam2', lam2, 'r1', 1e-2, 'r2', 1e-2);
    [mrl_X, mrl_W, mrl_iA, mrl_iB, ~] = MnPCA(td.Y, 2, opts);
    [U, D, ~] = gmd_method(td.Y, mrl_iA, mrl_iB, 2, 1e-4);
    mrl_score = U * D;
    mrl_pred = struct('signal', mrl_X * mrl_W', 'score', mrl_score, ...
        'iA', mrl_iA, 'iB', mrl_iB);
    subplot(2, 3, 3 * (idx - 1) + 2)
    plotScatter(mrl_pred.score, true.labels, palettes, sz);
    title('MN-PCA', 'fontSize', 10, 'FontWeight', 'Normal')
    if idx == 2
        xlabel("PC1")
    end
    drawnow;
    %% MN-PCA-w2
    [w2_Y, w2_iA, w2_iB] = MnPCAq1_wrapper(td.Y, 2, 0.05, 0.05, 1, td.Y - td.E, 500);
    w2_score = pca(w2_Y);
    w2_pred = struct('signal', w2_Y, 'score', w2_score, 'iA', w2_iA, 'iB', w2_iB);
    subplot(2, 3, 3 * (idx - 1) + 3);
    plotScatter(w2_pred.score, true.labels, palettes, sz);
    title('MN-PCA-w2', 'fontSize', 10, 'FontWeight', 'Normal')
    if idx == 2
        xlabel("PC1")
    end
    drawnow;
    %% plot figures  
end
% save figure to directory
saveas(gcf,'output/small_synthetic_projection.png')
```
