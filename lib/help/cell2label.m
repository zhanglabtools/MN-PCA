function [label, n_clust, names] = cell2label(cell_label)
% Convert cell_label to label
% [label, n_clust, names] = cell2label(cell_label)
names = unique(cell_label);
n_clust = length(names);
label = zeros(length(cell_label), 1);
for idx = 1:n_clust
    label(strcmp(names{idx}, cell_label)) = idx;
end
end

