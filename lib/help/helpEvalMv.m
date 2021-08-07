function metrics = helpEvalMv(eval_names, score, label, rep)
% Help function to evaluate the output of matrix.
% metrics = helpEvalMv(eval_names, score, label, rep)
%   eval_names: cells of evaluation names
%   score
%   label: true labels.
%   rep: number of replications.
nfold = 5;
metrics = zeros(length(eval_names), rep);
n_clust = length(unique(label));
if isscalar(score)
    return;
end
for i = 1:length(eval_names)
    if strcmp(eval_names{i}, 'Kmeans')
        metrics(i, :) = kmeansEval(n_clust, score, label, rep);
    elseif strcmp(eval_names{i}, 'KNN')
        metrics(i, :) =  knnEval(nfold, score, label, rep);
    elseif strcmp(eval_names{i}, 'KNN-LOO')
        metrics(i, :) =  knnEval(size(score, 1), score, label, rep);        
    elseif strcmp(eval_names{i}, 'LSSVM')
        metrics(i, :) =  lssvmEval(nfold, score, label, rep);            
    else
        error('Metric %s not implemented\n', eval_names{i});
    end
end
end

