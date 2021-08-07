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
REP = 5;
n_iter_v = [300, 500, 500, 500, 500];
lam1_v = [.5, .5, .8, .8, .8];
lam2_v = [.5, .5, .8, .9, 1.0];
lamr_v = [0.05, 0.05, 0.05, 0.05, 0.05];
%% Construct function to gen stru to save metrics.
genStr = genStrFun(REP, 'frob', 'psnr', 'rmse', 'nmi', 'prec1','prec2',...
                   'tpr1', 'tpr2', 'tnr1', 'tnr2', 'time'); 
LL_res = genStr();
LS_res = genStr();
SL_res = genStr();
SS_res = genStr();

for idx_p =  2:length(p_v)
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
    n_iter = n_iter_v(idx_p);
    dpath = "td.mat";
    out_path = "temp/gpca_res.mat";
    % struct to save results
    for idx = 1:REP
        td = toy_data2(populations, centroids,  0, opts_td);
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
    % save to mat file
    fprintf('save file to ./output/large_syn/gpca_p=%d.mat\n', p_v(idx_p));
    save(sprintf('./output/large_syn/gpca_p=%d.mat', p_v(idx_p)), 'LL_res',  'LS_res', 'SL_res', 'SS_res');
end
%% Print table
p_v = [2000, 3000, 4000, 5000, 6000];
metrics_idx = [2, 4];
method_names = {'LL', 'LS', 'SL', 'SS'};
for idx_p = 1:length(p_v)
    res = load(sprintf('./output/large_syn/gpca_p=%d.mat', p_v(idx_p)));
    fprintf('==========%d\n', p_v(idx_p));
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
