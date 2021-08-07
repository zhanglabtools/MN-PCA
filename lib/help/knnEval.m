function acc = knnEval(num_fold, score, labels, rep, n_neighbor)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
if ~exist('n_neighbor', 'var')
    n_neighbor = 3;
end
acc = zeros(rep, 1);
for k = 1:rep
    mdl = fitcknn(score, labels, 'NumNeighbors', n_neighbor); 
    cvmodel = crossval(mdl, 'KFold', num_fold);
    acc(k) = (1 - cvmodel.kfoldLoss) * 100;
end
end

