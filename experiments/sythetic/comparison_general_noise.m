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
rc_v = 1 ./[32, 64];
REP = 10;
spa = 0.01;
%% Construct function to gen stru to save metrics.
genStr = genStrFun(REP, 'frob', 'psnr', 'rmse', 'nmi', 'prec1','prec2',...
                   'tpr1', 'tpr2', 'tnr1', 'tnr2');
noise_types = {'uniform', 'mixture1', 'mixture2', 'mixture3'};
dpath = 'td_general_noise.mat';
out_path = 'temp/gpca_res.mat';
for idx_rc = 2:length(rc_v)
for i = 1:length(noise_types)
    opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc_v(idx_rc),...
                  'rc_B', rc_v(idx_rc));
    fprintf('Noise type: %s\n', noise_types{i});
    td = toy_data_general(populations, centroids, sig, opts_td, noise_types{i});
    [cand_lam1, cand_lam2] = cand_lam(td.Y, 2, .05, 3, 20);
    [lam1, lam2, score1, score2] = choose_lam(td.Y, 2, cand_lam1, cand_lam2);
    pca_res = genStr();
    LL_res = genStr();
    LS_res = genStr();
    SL_res = genStr();
    SS_res = genStr();
    mnpca_res = genStr();
    w2_res = genStr();
    for idx = 1:REP
        rng(idx);
        td = toy_data_general(populations, centroids, sig, opts_td, noise_types{i});
        X = td.Y;
        save(dpath, 'X');
        true = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, 'nclust', ...
            3, 'labels', td.labels);
        opts = struct('tol', 1e-3, 'est_cov', 'mle', 'lam1', lam1, 'lam2', lam2, ...
            'r1', 1e-2, 'r2', 1e-2);
        % perform PCA
        [pca_pred.score, pca_map] = pca(td.Y);
        pca_pred = struct('signal', pca_pred.score * pca_map.M' + repmat(pca_map.mean, n, 1), ...
            'score', pca_pred.score);
        pca_res.evaluation(idx, 1:4) = helpEval(pca_res.metrics_names(1:4), true, pca_pred); 
        % perform GPCA
        system(sprintf('Rscript runGPCA.R %s %s %d', dpath, out_path, dim));
        if isfile("temp/gpca_res.mat")
            gpca_res = load("temp/gpca_res.mat" );
            LL_pred = struct('signal', gpca_res.LL_score * diag(gpca_res.LL_D) *  gpca_res.LL_V',  ...
            'score', gpca_res.LL_score);
            LS_pred = struct('signal', gpca_res.LS_score  * diag(gpca_res.LS_D) *  gpca_res.LS_V',  ...
            'score', gpca_res.LS_score);
            SL_pred = struct('signal', gpca_res.SL_score  * diag(gpca_res.SL_D) *  gpca_res.SL_V',  ...
            'score', gpca_res.SL_score);
            SS_pred = struct('signal', gpca_res.SS_score * diag(gpca_res.SS_D) *  gpca_res.SS_V',  ...
            'score', gpca_res.SS_score);
        if ~any(isnan(LL_pred.score(:)))
            LL_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, LL_pred);
        end
        if ~any(isnan(LS_pred.score(:)))
            LS_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, LS_pred);
        end
        if ~any(isnan(SL_pred.score(:)))
            SL_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, SL_pred);
        end
        if ~any(isnan(SS_pred.score(:)))
            SS_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, SS_pred);  
        end
            delete('temp/gpca_res.mat');
        end
        % perform MN-PCA
        [mnpca_X, mnpca_W, mnpca_iA, mnpca_iB, ~] = MnPCA(td.Y, dim, opts);
        [U, D, ~] = gmd_method(td.Y, mnpca_iA, mnpca_iB, dim, 1e-4);
        mnpca_score = U * D;
        mnpca_pred = struct('signal', mnpca_X * mnpca_W', 'score', mnpca_score,...
            'iA', mnpca_iA, 'iB', mnpca_iB); 
        mnpca_res.evaluation(idx, :) = helpEval(mnpca_res.metrics_names, true, mnpca_pred);
        % perform MN-PCA w2
        [Y, iA, iB] = MnPCAq1_wrapper(td.Y, 2, lam1, lam2, 1, td.Y - td.E, 500);
        w2_score = pca(Y);
        w2_pred = struct('signal', Y, 'score', w2_score, 'iA', iA, 'iB', iB);
        w2_res.evaluation(idx, :) = helpEval(w2_res.metrics_names, true, w2_pred);
    end
    fprintf('save file to ./output/general_noise/comparison_res_c=%d-%s.mat\n', ...
        1/ rc_v(idx_rc), noise_types{i});
    save(sprintf('./output/general_noise/comparison_res_c=%d-%s.mat', ...
        1/ rc_v(idx_rc), noise_types{i}), 'pca_res', 'LL_res', 'LS_res', ...
        'SL_res', 'SS_res', 'mnpca_res', 'w2_res');
end 
end

%% Print tables
clear
shown_methods = [3, 1, 2, 4];
rc_v = 1 ./[32, 64];
rc = 1 / 32;
noise_types = {'Laplace', 'uniform', 'mixture1', 'mixture2', 'mixture3'};
method_names = {'SS_res', ...
    'mnpca_res', 'pca_res', 'w2_res'};
for idx_rc = 2:length(rc_v)
    fid = fopen(sprintf('./output/general_noise/comparison_general_noise_c=%d.txt', ...
        1 / rc_v(idx_rc)), 'w+');
    fprintf(fid, '==================c=%d====================\n', 1 / rc_v(idx_rc));
    for i = 1:length(noise_types)
        fprintf(fid, '-------------Noise Type=%s----------------\n', noise_types{i});
        res = load(sprintf('./output/general_noise/comparison_res_c=%d-%s.mat', ...
            1/ rc_v(idx_rc), noise_types{i}));
        fprintf(fid, 'PSNR\t'); 
        for method_idx = shown_methods
            method_res = res.(method_names{method_idx});
            
            fprintf(fid, '&%.2f(%.2f)', ...
                    mean(method_res.evaluation(:, 2)), std(method_res.evaluation(:, 2)));
        end
        fprintf(fid, '\n');
        fprintf(fid, 'NMI\t'); 
        for method_idx = shown_methods
            method_res = res.(method_names{method_idx});
            fprintf(fid, '&%.2f(%.2f)', ...
            mean(method_res.evaluation(:, 4)), std(method_res.evaluation(:, 4)));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end
