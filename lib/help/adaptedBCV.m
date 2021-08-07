function [dim, R] =  adaptedBCV(Y, upper_k, th)
% Adapted bicross validation to choose the aprriaote dimension
% 
    [~, D] = svds(Y, min(upper_k, min(size(Y))));
    d = diag(D);
    d = d.^2;
    R = cumsum(d) / sum(d);
    dim = find(R > th, 1);
    dim = min(dim, size(Y, 2) - 2);
    if dim < 2 
        dim = 2;
    end
    if size(Y, 2) <= 15
        dim = 2;
    end
end
