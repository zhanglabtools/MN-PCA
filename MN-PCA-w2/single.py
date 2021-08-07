# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:25:50 2019

@author: admin
"""
import MNPCA_W2 as mnpca
# import toydata_conditioned as td
import time
import numpy as np
import scipy.io as scio
def main():
    t = time.clock()
    ROWNUM = 200
    COLNUM = 150
    nv = 1
    sparsity = 0.98
    
    Lambda = 5
    lamdA = Lambda
    lamdB = Lambda
    condition_number = 1/32
    noise_variance = nv
    # toydata,truedata = td.generate_toydata(ROWNUM,COLNUM,noise_variance,sparsity,condition_number)
    # path1 = 'C:\\Users\\admin\\Documents\\toydata0.5.mat'
    # path2 = 'C:\\Users\\admin\\Documents\\truedata0.5.mat'
    # scio.savemat(path1,{'toydata':toydata})
    input_data = scio.loadmat('./temp/in.mat')
    Y = input_data['Y']
    rank = 2
    prins,A,B,stats = mnpca.W2(Y, np.zeros_like(Y), lamdA,lamdB,noise_variance,rank)
        
    print(time.clock()-t)
if __name__ == "__main__":
    t = time.clock()
    ROWNUM = 200
    COLNUM = 150
    nv = 1
    sparsity = 0.98
    
    Lambda = 5
    lamdA = Lambda
    lamdB = Lambda
    condition_number = 1/32
    noise_variance = nv
    # toydata,truedata = td.generate_toydata(ROWNUM,COLNUM,noise_variance,sparsity,condition_number)
    # path1 = 'C:\\Users\\admin\\Documents\\toydata0.5.mat'
    # path2 = 'C:\\Users\\admin\\Documents\\truedata0.5.mat'
    # scio.savemat(path1,{'toydata':toydata})
    input_data = scio.loadmat('./temp/in.mat')
    Y = input_data['Y']
    rank = 2
    prins,A,B,stats = mnpca.W2(Y, np.random.randn(*Y.shape), lamdA,lamdB,noise_variance,rank)
        
    print(time.clock()-t)
