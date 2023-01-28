# The basic usage of MN-PCA
Matrix normal PCA provides two algorithms to obtain low-rank representaion and the two-way noise structure.

First add folders to the working path.
``` matlab
addpath(genpath('./lib'));
addpath('MN-PCA-w2/');
addpath('MN-PCA-MRL/');
```
Generate the toy data and visualize it.
``` matlab
populations = [100, 100, 100];
n = sum(populations);
p = 200;
c = length(populations);
centroids = zeros(c, p);
l = 20;
rng('default')
centroids(1, 1:l) = 1;
centroids(1, end-l:end) = 1;
centroids(2, 1:l) = -1;
centroids(2, end-l+1:end) = -1;
centroids(3, 1:l) = 1;
centroids(3, end-l+1:end) = -1;
dim = 2;
scale = 1;
sig = 0;
rc_v = 1 ./[32, 192];
spa = 0.01;
sz = 6;
set_fig('units','inches','width', 7.0,'height', 3.75,'font','Times New Roman','fontsize', 8)
palettes = cbrewer('qual', 'Set1', 3); 
```


