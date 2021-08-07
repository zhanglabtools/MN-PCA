function res_dir = run_experiment(X, label, dim, output_dir, opts)
%routine function for experiments
%   X: n*p data  matrix
%   label: sample label. Integer vector.
%   dim: dimension
%   out_dir: output dir
%   opts: optoins struct contains following fields
%     .rep: replicates of evaluation
%     .all_methods: binary. run all competing methods
REP = opts.rep;
n_clust = length(unique(label));
gen_mvres = genStrFun(REP, 'Kmeans', 'KNN', 'LSSVM');
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end
fprintf('Data n=%d p=%d nclust=%d\n', size(X, 1), size(X, 2), length(unique(label)));
%% Choose lambda
out_prefix = strcat(output_dir, 'dim=', num2str(dim));
if isfield(opts, 'prop1') && isfield(opts, 'prop2')
    if opts.prop1 < 0
        cand_lam1 = [];
    else
        cand_lam1 = cand_lam(X, dim, opts.prop1, 3, 20);
    end
    if opts.prop2 < 0
        cand_lam2 = [];
    else
        [~, cand_lam2] = cand_lam(X, dim, opts.prop2, 3, 20);
    end
else
    [cand_lam1, cand_lam2] = cand_lam(X, dim, .1, 3, 20);
end
[lam1, lam2, ~, ~] = choose_lam(X, dim, cand_lam1, cand_lam2);
fprintf('Choosen lam1=%.2f lam2=%.2f\n', lam1, lam2);
%% run MN-PCA\
rng('default')
rng(0);
mnff_opts = struct('tol', 1e-3, 'est_cov', 'mle', 'lam1', lam1, 'lam2', lam2, 'r1', 1e-2, 'r2', 1e-2, 'max_iter', opts.max_iter);
tic;
[mnff_X, mnff_W, mnff_iA, mnff_iB, mnff_Out] = MnPCA(X, 2, mnff_opts);
mnff_rt = toc;
[U, D, V] = gmd_method(X, mnff_iA, mnff_iB, dim, 1e-4);
mnff_score = U * D;
% run PCA
tic;
pca_score = pca(X, dim);
pca_rt = toc;
rt_v = [pca_rt, mnff_rt];   
if opts.run_all
    % run PPCA
    tic;
    [~,ppca_score, ~] = ppca(X,dim);
    ppca_rt = toc; 
    % gplvm
    tic;
    gplvm_score = gplvm(X, dim, 1);
    gplvm_rt = toc;
    % kernel_pca
    tic;
    [kpca_lin_score, ~] = kernel_pca(X, dim, 'linear');
    kpca_lin_rt = toc;
    tic;
    [kpca_gau_score, ~] = kernel_pca(X, dim, 'gauss');
    kpca_gau_rt = toc;
    tic;
    [kpca_pol_score, ~] = kernel_pca(X, dim, 'poly');
    kpca_pol_rt = toc;
    % fast ICA
    tic;
    ica_score = fastICA(X', dim);
    ica_rt = toc;
    ica_score = ica_score';
    % gpca
    tic;
    D = pdist(X, 'chebychev');
    ratio = max(1e-4, nnz(mnff_iA) / numel(mnff_iA));
    window = quantile(D, ratio);
    D = squareform(D);
    pre_rt = toc;
    [Q_S, is_sp1] = expSmooth(D, 2);
    try 
        [Q_L, is_sp2] = lapDist(D, window);
    catch
        Q_L = 0; is_sp2 = 0;
    end
    % prepare among-column matrix
    D = pdist(X', 'chebychev');
    ratio = max(1e-4, nnz(mnff_iB) / numel(mnff_iB));
    window = quantile(D, ratio);
    D = squareform(D);
    [R_S, is_sp3] = expSmooth(D, 2);
    try
        [R_L, is_sp4] = lapDist(D, window);
    catch
        R_L = 0; is_sp4 = 0;
    end
    % run gpca
    % run GPCA(SS)
    if is_sp1 && is_sp3
        tic;
        [U, D, ~] = gmd_method(X, Q_S, R_S, dim, 1e-4);
        SS_score = U * D;
        gpca_ss_rt = toc + pre_rt;
    else
        SS_score = 0;
        gpca_ss_rt = 0;
    end
    % run GPCA (SL)
    if is_sp1 && is_sp4
        tic;
        [U, D, ~] = gmd_method(X, Q_S, R_L, dim, 1e-4);
        SL_score = U * D;
        gpca_sl_rt = toc + pre_rt;
    else
        SL_score = 0;
        gpca_sl_rt = 0;
    end
    if is_sp2 && is_sp3
        tic;
        [U, D, ~] = gmd_method(X, Q_L, R_S, dim, 1e-4);
        LS_score = U * D;
        gpca_ls_rt = toc + pre_rt;
    else
        LS_score = 0;
        gpca_ls_rt = 0;
    end
    if is_sp1 && is_sp4
        tic;
        [U, D, ~] = gmd_method(X, Q_L, R_L, dim, 1e-4);
        LL_score = U * D;
        gpca_ll_rt = toc + pre_rt;
    else
        LL_score = 0;
        gpca_ll_rt = 0;
    end
    % PPCA
    rt_v = [pca_rt, gplvm_rt, kpca_lin_rt, kpca_gau_rt, kpca_pol_rt, ica_rt, ...
                  gpca_ll_rt, gpca_ls_rt, gpca_sl_rt, gpca_ss_rt, ...
                  mnff_rt, ppca_rt];
end
%% Evaluation 
% MN-PCA
mnff_mvres = gen_mvres();
mnff_mvres.evaluation = helpEvalMv(mnff_mvres.metrics_names, mnff_score, label, REP);
% PCA
pca_mvres = gen_mvres();
pca_mvres.evaluation = helpEvalMv(pca_mvres.metrics_names, pca_score, label, REP);
% PPCA

if opts.run_all
    % PPCA
    ppca_mvres = gen_mvres();
    ppca_mvres.evaluation = helpEvalMv(ppca_mvres.metrics_names, ppca_score, label, REP);
    % GPLVM
    gplvm_mvres = gen_mvres();
    gplvm_mvres.evaluation = helpEvalMv(gplvm_mvres.metrics_names, gplvm_score, label, REP);
    % KPCA
    kpca_lin_mvres = gen_mvres();
    kpca_lin_mvres.evaluation = helpEvalMv(kpca_lin_mvres.metrics_names, kpca_lin_score, label, REP);
    kpca_gau_mvres = gen_mvres();
    kpca_gau_mvres.evaluation = helpEvalMv(kpca_gau_mvres.metrics_names, kpca_gau_score, label, REP);
    kpca_pol_mvres = gen_mvres();
    kpca_pol_mvres.evaluation = helpEvalMv(kpca_pol_mvres.metrics_names, kpca_pol_score, label, REP);
    % Fast-ICA
    ica_mvres = gen_mvres();
    ica_mvres.evaluation = helpEvalMv(ica_mvres.metrics_names, ica_score, label, REP);
    % GPCA
    if isscalar(LL_score) || any(isnan(LL_score(:)))
        gpca_ll_mvres = 0;
    else
        gpca_ll_mvres = gen_mvres();
        gpca_ll_mvres.evaluation = helpEvalMv(gpca_ll_mvres.metrics_names, LL_score, label, REP);
    end

    if isscalar(LS_score) || any(isnan(LS_score(:)))
        gpca_ls_mvres = 0;
    else
        gpca_ls_mvres = gen_mvres();
        gpca_ls_mvres.evaluation = helpEvalMv(gpca_ls_mvres.metrics_names, LS_score, label, REP);
    end

    if isscalar(SL_score) || any(isnan(SL_score(:)))
        gpca_sl_mvres = 0;
    else
        gpca_sl_mvres = gen_mvres();
        gpca_sl_mvres.evaluation = helpEvalMv(gpca_sl_mvres.metrics_names, SL_score, label, REP);
    end

    if isscalar(SL_score) || any(isnan(SL_score(:)))
        gpca_ss_mvres = 0;
    else
        gpca_ss_mvres = gen_mvres();
        gpca_ss_mvres.evaluation = helpEvalMv(gpca_ss_mvres.metrics_names, SS_score, label, REP);
    end     
end

if opts.run_all
    mvres_cell = {pca_mvres, gplvm_mvres, kpca_lin_mvres, kpca_gau_mvres, kpca_pol_mvres, ica_mvres, ...
                  gpca_ll_mvres, gpca_ls_mvres, gpca_sl_mvres, gpca_ss_mvres, ...
                  mnff_mvres, ppca_mvres};
    competing_methods_cell = {'PCA', 'GPLVM', 'KPCA(lin)', 'KPCA(gau)', 'KPCA(pol)', 'ICA', ...
                             'GPCA(LL)', 'GPCA(LS)', 'GPCA(SL)', 'GPCA(SS)', ...
                             'MN-PCA', 'PPCA'};
else
    mvres_cell = {pca_mvres, mnff_mvres};
    competing_methods_cell = {'PCA', 'MN-PCA'};
end
fprintf('Total number of results: %d.\n', length(mvres_cell));
 %% print table
if isfield(opts, 'table_fid')
    fid = opts.table_fid;
else
    fid = fopen(strcat(out_prefix, 'table.txt'), 'w');
end
fprintf(fid, 'Dataset:%s #sample=%d #feature=%d #class=%d\n', opts.dname, size(X, 1), size(X, 2), n_clust);
fprintf(fid, 'dim = %d\n', dim);
fprintf(fid, 'MN-PCA Setting: lam1=%.4f lam2=%.4f ', lam1, lam2);
fprintf(fid, '\n');
fprintf(fid, 'Evaluation\t');
fprintf(fid, '%s\t', competing_methods_cell{:});
fprintf(fid, '\n');
% print Kmeans
fprintf(fid, 'Kmeans (nmi)\t');
for mvres_idx = 1:length(mvres_cell)
    mvres = mvres_cell{mvres_idx};
    if ~isstruct(mvres)
       fprintf(fid, 'NA\t'); 
    else
        fprintf(fid, '%.2f(%.2f)\t', mean(mvres.evaluation(1, :)), std(mvres.evaluation(1, :)));
    end
    
end
fprintf(fid, '\n');
% printf KNN
fprintf(fid, 'KNN (acc)\t');
for mvres_idx = 1:length(mvres_cell)
    mvres = mvres_cell{mvres_idx};
    if ~isstruct(mvres)
       fprintf(fid, 'NA\t'); 
    else    
        fprintf(fid, '%.2f(%.2f)\t', mean(mvres.evaluation(2, :)), std(mvres.evaluation(2, :)));
    end
end
fprintf(fid, '\n');
% printf LSSVM
fprintf(fid, 'LSSVM (acc)\t');
for mvres_idx = 1:length(mvres_cell)
    mvres = mvres_cell{mvres_idx};
    if ~isstruct(mvres)
       fprintf(fid, 'NA\t'); 
    else    
        fprintf(fid, '%.2f(%.2f)\t', mean(mvres.evaluation(3, :)), std(mvres.evaluation(3, :)));
    end
end
fprintf(fid, '\n');
fprintf(fid, 'Time (s)\t');
for idx = 1:length(mvres_cell)
    fprintf(fid, '%.2f\t', rt_v(idx));
end
if fid ~= 1
    fclose(fid);
end
%% Save results
res_dir = strcat(out_prefix, 'res.mat');
save(res_dir, 'mnff_score', 'mnff_X','mnff_W', 'mnff_iA', 'mnff_iB', 'mnff_Out', 'label', 'mvres_cell', 'dim');

end

