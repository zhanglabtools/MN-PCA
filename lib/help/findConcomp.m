function [label, group] = findConcomp(D, th, min_sz)
%Find connected componet in a distance matrix.
% [label, order] = FINDCONCOMP(D, th)
% th thresholding
% if ~issymmetric(D)
%     error('Distance matrix must be symmtric.\n');
% end
n = size(D, 1);
label = zeros(1, n);
D_ = abs(D - diag(diag(D))) > th;
cur_label = 1;
visited = zeros(1, n);
for node = 1:n
    if ~visited(node)
        visited(node) = 1;
        label(node) = cur_label;
        [neighbors, visited] = dfs(node, D_, visited, []);
        label(neighbors) = cur_label;
        cur_label = cur_label + 1;
    end
end
% res = tabulate(label);
% iso_label = find(res(:, 2)==1);
% ulabel = unique(label);
% ncomps = length(ulabel) - length(iso_label) + 1;
% cur_label = 1;
% for idx = ulabel
%     if ismember(idx, iso_label) 
%         label(label==idx) = ncomps;
%     else
%         label(label==idx) = cur_label;
%         cur_label = cur_label + 1;
%     end
% end
group = merge(label, min_sz);
end

function [neighbors, visited] = dfs(node, D_, visited, neighbors)
    node_list = find(D_(node, :));
    for nodei = node_list
        if ~visited(nodei)
            neighbors = [neighbors; nodei]; %#ok<AGROW>
            visited(nodei) = 1;
            [neighbors, visited] = dfs(nodei, D_, visited, neighbors);
        end
    end           
end

function group = merge(label, min_size)
    res = tabulate(label);
    group = {[]};
    active = 1;
    for idx = 1:size(res, 1)
        if res(idx, 2) >= min_size
            group = {idx; group}; %#ok<AGROW>
            active = active + 1;
        else
            if length(group) < active
                group{active} = [];
            end
            group{active} = [group{active}, idx];
            if length(group{active}) >= min_size
                active = active + 1;
            end
        end
    end
end