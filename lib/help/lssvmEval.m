function [acc, model] = lssvmEval(nfold, score, label, rep)
%Evaluation with LSSVM
%   Detailed explanation goes here
warning('off'); %#ok<*WNOFF>
acc = zeros(rep, 1);
num_obs = length(label);
rng('default')
rng(0);
model = initlssvm(score, label, 'classifier', [], [], 'RBF_kernel');
model = tunelssvm(model, 'simplex', 'crossvalidatelssvm', {nfold, 'misclass'},'code_OneVsAll');
for i = 1:rep
    folds = cvpartition(num_obs, 'KFold', nfold);
    cv_acc = zeros(nfold, 1);
    for idx_fold = 1:nfold
        cvmodel = trainlssvm(model, score(folds.training(idx_fold), :),  label(folds.training(idx_fold)));
        pred = simlssvm(cvmodel, score(folds.test(idx_fold), :));
        cv_acc(idx_fold) = sum(pred == label(folds.test(idx_fold))) / length(pred);
    end
    acc(i) = mean(cv_acc);
end
warning('on'); %#ok<*WNON>
acc = acc * 100;
model = trainlssvm(model);
end

