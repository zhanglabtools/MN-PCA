library('shock')
library('R.Matlab')

## load data to test 
data(dataTest)

## dimension of the dataset expdata
n <- dim(dataTest)[1]
p <- dim(dataTest)[2]

## perform partition of variables selection
## based on the slope heuristic
resShock <- shockSelect(dataTest)


## verify that the two slope heuristic 
## calibrations give the same result
table(resShock$SHDJlabels == resShock$SHRRlabels)

## collect the labels of variables 
SHlabels  <- resShock$SHDJlabels
## Residual matrix
res <- readMat('../../residual.mat')
R = res$R;
resShock <- shockSelect(R)

