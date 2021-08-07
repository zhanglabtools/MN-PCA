function nmi = kmeansEval(n_clust, score, labels, rep)
% KMEANSEVAL perf = KMEANSEVAL(n_cluster, score, labels, rep)
% kmeans to evaluate the clustering performance. 
% Return a struct contains following metrics: NMI, purity.
% 
%     Input:
%       n_cluster -- number of  clusters
%       score     -- number of samples * d matrix
%       labels    -- ground truth labels
%     Output:
%       perf --- performance structure with fields
%            nmi     -- normalized mutual information
%            purity  -- purity
%idx = int8(idx);
%labels = int8(labels);
if ~exist('rep', 'var')
    rep = 100;
end
nmi = zeros(rep, 1);
%perf.purity = zeros(rep, 1);
for k = 1:rep
    idx = kmeans(score, n_clust, 'MaxIter',1000);
    nmi(k) = cal_nmi2(labels, idx) * 100;
    %perf.purity(k) = cal_purity(labels, idx) * 100;
end
end

%% local function

function purity = cal_purity(true_labels, pred_labels)
cfmat = confusionmat(true_labels, pred_labels);
purity = sum(max(cfmat, [], 2)) / length(true_labels);
end

function nmi = cal_nmi2(x, y)
% this function borrowed from 
% https://www.mathworks.com/matlabcentral/fileexchange/29047-normalized-mutual-information
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x), min(y));
x = x-l+1;
% class(y(1))
% class(l(1))
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));


% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));

% mutual information
MI = Hx + Hy - Hxy;

% normalized mutual information
nmi = sqrt((MI/Hx)*(MI/Hy));
nmi = max(0,nmi);
end