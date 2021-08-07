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
%% Construct function to gen stru to save metrics.
genStr = genStrFun(REP, 'frob', 'psnr', 'rmse', 'nmi', 'prec1','prec2',...
                   'tpr1', 'tpr2', 'tnr1', 'tnr2', 'time'); 
LL_res = genStr();
LS_res = genStr();
SL_res = genStr();
SS_res = genStr();
for i = 1:length(rc_v)
    opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc_v(i),...
                  'rc_B', rc_v(i));
    dpath = "td.mat";
    out_path = "temp/gpca_res.mat";
    % struct to save results
    for idx = 1:REP
        td = toy_data2(populations, centroids, sig, opts_td);
        X = td.Y;
        true = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, 'nclust', 3, 'labels', td.labels);
        save(dpath, 'X')
        system(sprintf('Rscript runGPCA.R %s %s %d', dpath, out_path, dim));
        gpca_res = load("temp/gpca_res.mat" );
        [pca_pred.score, pca_map] = pca(td.Y);
        pca_pred = struct('signal', pca_pred.score * pca_map.M' + repmat(pca_map.mean, n, 1), 'score', pca_pred.score);
        LL_pred = struct('signal', gpca_res.LL_score * diag(gpca_res.LL_D) *  gpca_res.LL_V',  ...
        'score', gpca_res.LL_score);
        LS_pred = struct('signal', gpca_res.LS_score  * diag(gpca_res.LS_D) *  gpca_res.LS_V',  ...
        'score', gpca_res.LS_score);
        SL_pred = struct('signal', gpca_res.SL_score  * diag(gpca_res.SL_D) *  gpca_res.SL_V',  ...
        'score', gpca_res.SL_score);
        SS_pred = struct('signal', gpca_res.SS_score * diag(gpca_res.SS_D) *  gpca_res.SS_V',  ...
        'score', gpca_res.SS_score);    
        % evaluations  
    	LL_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, LL_pred);
        LL_res.evaluation(idx, end) = gpca_res.LL_time;
        LS_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, LS_pred);
        LS_res.evaluation(idx, end) = gpca_res.LS_time;
        SL_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, SL_pred);
        SL_res.evaluation(idx, end) = gpca_res.SL_time;
        SS_res.evaluation(idx, 1:4) = helpEval(LL_res.metrics_names(1:4), true, SS_pred);
        SS_res.evaluation(idx, end) = gpca_res.SS_time;
    end
    % print results
    fprintf('save file to ./output/small_syn/gpca_res=%d.mat\n', 1/ rc_v(i));
    save(sprintf('./output/small_syn/gpca_res=%d.mat', 1/ rc_v(i)), 'LL_res',  'LS_res', 'SL_res', 'SS_res');% , 'other_res');
end
%% Print resutls to tables
metrics_idx = [2, 4];
method_names = {'LL', 'LS', 'SL', 'SS'};
for i = 1:length(rc_v)
    res = load(sprintf('./output/small_syn/gpca_res=%d.mat', 1/ rc_v(i)));
    fprintf('==========%d\n', 1/ rc_v(i));
    for j = 1:4
        method_res = res.(sprintf('%s_res', method_names{j}));
        fprintf("%s\t",method_names{j});
        for l = 1:length(metrics_idx)
            fprintf('%.2f(%.2f)\t', mean(method_res.evaluation(:, metrics_idx(l))), ...
            std(method_res.evaluation(:, metrics_idx(l))));
        end
        fprintf("\n");
    end
end
