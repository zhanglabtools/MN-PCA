function fh = plotScatter(score, label, palettes, sz, lim)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if ~exist('lim', 'var')
    lim = 1.1;
end
    
unique_label = unique(label);
assert( length(unique(label)) == size(palettes, 1));
hold on;
for idx = 1:length(unique_label)    
scatter(score(label == unique_label(idx), 1), score(label == unique_label(idx), 2), sz, palettes(idx, :), 'filled', 'o');
end
hold off;
coord_max = max(score);
coord_min = min(score);
xlim([coord_min(1), coord_max(1)] * lim);
ylim([coord_min(2), coord_max(2)] * lim);
fh = gca;
end

