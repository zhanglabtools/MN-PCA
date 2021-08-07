import argparse
import numpy as np
import scipy.io as scio
import MNPCA_Q1 as mnpca

parser = argparse.ArgumentParser(description='MnPCA q1 distance')
parser.add_argument('dpath', type=str, help='input data matrix string. `.mat`')
parser.add_argument('out', type=str, help='Output resutls directory')
parser.add_argument('dim', type=int, help='Dimension')
parser.add_argument('lam1', type=float, 
                    help='L1 regularization of among-row matrix')
parser.add_argument('lam2', type=float, 
                    help='L1 regularization of among-cols matrix')
parser.add_argument('-s', '--sigma', default=.5, type=float,
                    help='standard devivation of the whithe noise')
parser.add_argument('-n', '--n_iter', default=400, type=int)                    

args = parser.parse_args()
temp = scio.loadmat(args.dpath)
Y = temp['Y']
if 'true_Y' in temp:
    true_Y = temp['true_Y']
else:
    true_Y = Y
# print('args', args.n_iter, 'sigma', args.sigma)
prins, A, B, stats = mnpca.W2(Y, true_Y, args.lam1, args.lam2,
                              args.sigma, args.dim, args.n_iter)

scio.savemat(args.out, {'Y_': prins, 'iA': A, 'iB': B, })

