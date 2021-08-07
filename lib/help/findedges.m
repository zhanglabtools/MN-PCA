function [edges, values] = findedges(iA, type)
    iA_ = tril(iA, -1);
    [row, col] = find(iA_);
    idx = sub2ind(size(iA_), row, col);
    [values, order] = sort(iA_(idx), type);
    edges = [row(order), col(order)];
end