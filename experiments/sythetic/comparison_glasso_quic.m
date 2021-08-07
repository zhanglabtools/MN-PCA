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
for i = 1:length(rc_v)
    opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc_v(i),...
                  'rc_B', rc_v(i));
    td = toy_data2(populations, centroids, sig, opts_td);
    [cand_lam1, cand_lam2] = cand_lam(td.Y, 2, .05, 3, 20);
    [lam1, lam2, score1, score2] = choose_lam(td.Y, 2, cand_lam1, cand_lam2);
    gl_res = genStr();
    quic_res = genStr();
    for idx = 1:REP
        rng(idx);
        td = toy_data2(populations, centroids, sig, opts_td);
        true = struct('signal', td.Y - td.E, 'iA', td.iA, 'iB', td.iB, 'nclust', 3, 'labels', td.labels);
        opts = struct('tol', 1e-3, 'est_cov', 'mle', 'lam1', lam1, 'lam2', lam2, ...
            'r1', 1e-2, 'r2', 1e-2);
        % perform QUIC MN-PCA
        tic;
        [quic_X, quic_W, quic_iA, quic_iB, ~] = MnPCA(td.Y, dim, opts);
        quic_rt = toc;   
        [U, D, ~] = gmd_method(td.Y, quic_iA, quic_iB, dim, 1e-4);
        quic_score = U * D;
        quic_pred = struct('signal', quic_X * quic_W', 'score', quic_score,...
            'iA', quic_iA, 'iB', quic_iB);        
        quic_res.evaluation(idx, 1:end-1) = helpEval(quic_res.metrics_names(1:end-1), true, quic_pred);
        quic_res.evaluation(idx, end) = quic_rt;
        % perform Glasso MN-PCA
        tic;
        [gl_X, gl_W, gl_iA, gl_iB, ~] = MnPCAglasso(td.Y, dim, opts);
        gl_rt = toc;
        [U, D, ~] = gmd_method(td.Y, gl_iA, gl_iB, dim, 1e-4);
        gl_score = U * D;
        gl_pred = struct('signal', gl_X * gl_W', 'score', gl_score,...
            'iA', gl_iA, 'iB', gl_iB);
        gl_res.evaluation(idx, 1:end-1) = helpEval(gl_res.metrics_names(1:end-1), true, gl_pred);
        gl_res.evaluation(idx, end) = gl_rt;           
    end
    fprintf('save file to ./output/comparison_glasso/comparison_res_c=%d.mat\n', 1/ rc_v(i));
    save(sprintf('./output/comparison_glasso/comparison_res_c=%d.mat', ...
        1/ rc_v(i)), 'gl_res',  'quic_res');% , 'other_res');
end
% centroids = centroids * 1.2;
% centroids(3, 2 * l:end-l) = 1;
% rc_v = 1 ./[1, 4, 8, 12, 16];
% rc_v = 1 ./[8, 16, 32, 64, 96, 128, 160, 192, 224];
% rc = 1 / 192;
% REP = 10;
% spa = 0.01;
% opts_td = struct('scale', scale, 'alpha_A', spa, 'alpha_B', spa, 'rc_A', rc,...
%               'rc_B', rc);
% td = toy_data2(populations, centroids, sig, opts_td);
% [Uk, sk, W] = svds(td.Y, dim);
% X = Uk * sk;
% R = td.Y - X * W';
% lam = 0.1;
% [n, p] = size(td.Y); 
% iB = eye(n);
% S1 = estimate_cov(R', iB, 'mle');
% [iA1, theta, iter, avgTol, hasError] = myGLasso(S1, lam);
% [iA2, ~] = QUIC('default', S1, lam, 1e-4, 0, ...
%             20, iB, inv(iB));