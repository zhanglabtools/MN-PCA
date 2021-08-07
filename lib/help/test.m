n = 10;
x = randn(n, n);
D = pdist(x, 'chebychev');
D = squareform(D);
window = 2;
[L, issemi] = lapDist(D, window);