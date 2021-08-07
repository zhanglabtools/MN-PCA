function [lam1, lam2, score1, score2] = cv_lam(Y, dim, kfold1, cand_lam1, kfold2, cand_lam2)
%Using cross valiadation to choose the l1 value
%   [lam1, lam2, score1, score2] = CV_LAM(Y, dim, kfold1, cand_lam1, kfold2, cand_lam2)
[n, p] = size(Y);
[U, S, V] = svds(Y, dim);
R = Y - U * S * V'; % residual
% choose lam for rows
if isempty(cand_lam1)
    lam1 = -1;
    score1 = [];
else
    partition = cvpartition(p, 'KFold', kfold1); 
    score1 = zeros(size(cand_lam1));
    for i = 1:length(cand_lam1)
        score1(i) = cv_score(R', partition, cand_lam1(i));
    end
    [~, I] = min(score1);
    lam1 = cand_lam1(I);
end
% choose lam for columns
if isempty(cand_lam2)
    score2 = [];
    lam2 = -1;
else
    score2 = zeros(size(cand_lam2));
    partition = cvpartition(n, 'KFold', kfold2);
    for i = 1:length(cand_lam2)
        score2(i) = cv_score(R, partition, cand_lam2(i));
    end
    [~, I] = min(score2);
    lam2 = cand_lam2(I);
end
end

%% Nested function
function score = cv_score(R, partition, lam)
score = 0;
for idx = 1:partition.NumTestSets
    St = R(partition.training(idx), :);
    St = St' * St / (partition.TrainSize(idx) - 1);
    Sv = R(partition.test(idx), :);
    Sv = Sv' * Sv / (partition.TestSize(idx) - 1);
    [inv_St, ~] = QUIC('default', St, lam);
    score = score + trace(Sv * inv_St) - log(det(inv_St));
end
score = score / partition.NumTestSets;
end