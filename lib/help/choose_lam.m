function [lam1, lam2, score1, score2] = choose_lam(Y, dim, cand_lam1, cand_lam2)
%   Using cross valiadation to choose the l1 value
%   [lam1, lam2, score1, score2] = choose_lam(Y, dim, cand_lam1, cand_lam2)
if dim == 0
    R = Y;
else
    [U, S, V] = svds(Y, dim);
    R = Y - U * S * V'; % residual
end
% choose lam for rows
if isempty(cand_lam1)
    lam1 = -1;
    score1 = [];
else
    score1 = zeros(size(cand_lam1));
    for i = 1:length(cand_lam1)
        %score1(i) = cv_score(R', partition, cand_lam1(i));
        score1(i) = BIC(R', cand_lam1(i)); 
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
    for i = 1:length(cand_lam2)
        score2(i) = BIC(R, cand_lam2(i)); 
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
    score = score + trace(Sv * inv_St);
end
score = score / partition.NumTestSets;
end

function score = BIC(R, lam)
    n = size(R, 1);
    S = R' * R / (n - 1);
    [inv_S, ~] = QUIC('default', S, lam);
    val1 = trace(inv_S * S);
    val2 = -logdet(inv_S);
    val3 = nnz(inv_S);
    score = val1 + val2 + (val3 - n) * log(n) / n;
    fprintf('lam=%.2f|trace=%.2f -logdet=%.2f nnz=%.2f\n', lam, val1, val2, val3); 
end

function x = logdet(A)
L = chol(sparse(A), 'lower');
x =  sum(log(diag(L))) * 2;
x=full(x);
end
