function [num_edges, order_idx, palette] = plotPrec(iA, label, opts)
% num_edges = plotPrec(iA, label,  opts)
% Plot precision matrix redordered by label.
% opts .th         --- threholding value for entries in precision matrix
%      .mk_sz      --- size of markers
%      .scd_label  --- secondary label
palette = cbrewer('qual', 'Set1', max(length(unique(label)), 3), 'PCHIP');
if isfield(opts, 'scd_label')
    label_ = [label, opts.scd_label];
    [~, order_idx] = sortrows(label_, [1, 2]);
    order_label = label_(:, 1);
else
    [order_label, order_idx] = sort(label);
end
    
reorder_iA = abs(iA(order_idx, order_idx)) > opts.th;
num_edges = nnz(reorder_iA);
[row_idx, col_idx] = find(reorder_iA);
label_counts = tabulate(order_label);
label_counts = label_counts(:, 2);
stop = 0;
ploted_ind = 0;
hold on;
mk_sz = opts.mk_sz;
for idx = 1:length(unique(label))
    st = stop + 1;
    stop = st + label_counts(idx) - 1;
    ind = (row_idx <= stop & row_idx >= st) & (col_idx <= stop & col_idx >= st);
    scatter(row_idx(ind), col_idx(ind), mk_sz, palette(idx, :), 'filled');
    ploted_ind = ploted_ind + ind;
    
end

scatter(row_idx(~ploted_ind), col_idx(~ploted_ind), mk_sz * .5, [15 15 15]/255, 'filled');
stop = 0;
for idx = 1:length(unique(label))
    st = stop + 1;
    stop = st + label_counts(idx) - 1;
    ind = (row_idx <= stop & row_idx >= st) & (col_idx <= stop & col_idx >= st);
    rectangle('Position',[st st label_counts(idx)-.5 label_counts(idx) - .5], 'EdgeColor','r', 'LineWidth', 2)
    ploted_ind = ploted_ind + ind;
end
box on;
set(gca,'YDir','rev')
set(gca, 'XAxisLocation', 'top')
n = size(iA, 1);
xlim([1, n + .5])
ylim([1, n + .5])
end

