function metrics = evalPREC(true, pred)
%EVALPREC evalute the prediceted precicision matrix
% metrics = evalPREC(true, pred)
k = 150; % only conver about top100
n = size(true, 1);
%% for test
% true = td.iA;
% pred = iA;
%%
threshold_pred = topk_prec(pred, k);
threshold_true = topk_prec(true, k);
eps = 1e-5;
tp = sum(sum(and(threshold_pred > eps, threshold_true > eps))) - n;
tp = tp / 2;
fp = sum(sum(and(threshold_pred > eps, threshold_true < eps))) / 2;
tn = sum(sum(and(threshold_pred < eps, threshold_true < eps))) / 2;
fn = sum(sum(and(threshold_pred < eps, threshold_true > eps))) / 2;
metrics.recall = tp / (fn + tp);% = tpr
metrics.precision = tp / (fp + tp); 
metrics.tpr = tp / (fn + fp);
metrics.tnr = tn / (tn + fp);
end

function iA_ = topk_prec(iA, k)
iA_ = abs(tril(iA, -1));
vals = sort(abs(nonzeros(iA_)), 'descend');
if k > length(vals)
    th = 1e-5;
else
    th = vals(k);
end
iA_ = abs(iA) >= th;
end
