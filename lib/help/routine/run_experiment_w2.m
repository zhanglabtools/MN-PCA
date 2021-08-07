function res_dir = run_experiment_w2(X, label, dim, output_dir, opts)

REP = opts.rep;
n_clust = length(unique(label));
gen_mvres = genStrFun(REP, 'Kmeans', 'KNN', 'LSSVM');
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end
if isfield(opts, 'n_iter')
    n_iter = opts.n_iter;
else
    n_iter = 500;
end
fprintf('Data n=%d p=%d nclust=%d\n', size(X, 1), size(X, 2), length(unique(label)));
out_prefix = strcat(output_dir, 'w2dim=', num2str(dim));
%% run MN-PCA-w2
rng('default')
rng(0);
[X_, w2_iA, w2_iB] = MnPCAw2_wrapper(X, dim, opts.lam1, opts.lam2, opts.sig, randn(size(X)), n_iter); %#ok<ASGLU>
w2_score = pca(X_, dim);
%% evaluation
w2_mvres = gen_mvres();
w2_mvres.evaluation = helpEvalMv(w2_mvres.metrics_names, w2_score, label, REP);
%% print table
if isfield(opts, 'table_fid')
    fid = opts.table_fid;
else
    fid = fopen(strcat(out_prefix, 'table.txt'), 'w');
end
fprintf(fid, 'Dataset:%s #sample=%d #feature=%d #class=%d\n', opts.dname, size(X, 1), size(X, 2), n_clust);
fprintf(fid, 'dim = %d\n', dim);
fprintf(fid, 'MN-PCA Setting: lam1=%.2f lam2=%.2f ', opts.lam1, opts.lam2);
fprintf(fid, '\n');
fprintf(fid, 'Evaluation\t');
fprintf(fid, 'MN-PCA-w2');
fprintf(fid, '\n');
fprintf(fid, 'Kmeans (nmi)\t %.2f(%.2f)\n',  mean(w2_mvres.evaluation(1, :)), std(w2_mvres.evaluation(1, :)));
fprintf(fid, 'KNN (acc)\t %.2f(%.2f)\n',  mean(w2_mvres.evaluation(2, :)), std(w2_mvres.evaluation(2, :)));
fprintf(fid, 'LSSVM (acc)\t %.2f(%.2f)\n',  mean(w2_mvres.evaluation(3, :)), std(w2_mvres.evaluation(3, :)));
fclose(fid);
res_dir = strcat(out_prefix, 'res.mat');
save(res_dir, 'w2_score', 'X_',  'w2_iA', 'w2_iB',  'label', 'dim', 'w2_mvres');
end