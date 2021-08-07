# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:07:03 2019

@author: admin
"""
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import svd_numpy as svd
class Linear1(Module):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input,choice=None):
        if choice is not None:
            return F.linear(input, self.weight[choice,:], self.bias)
        else:
            
            return F.linear(input, self.weight,self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
class train_data():
    
    def __init__(self, colume, number):
        self.colume = colume
        self.number = number

    def __getitem__(self, index):#返回的是tensor
        img, target = self.colume[index], self.number[index]
        return img, target

    def __len__(self):
        return len(self.colume)
class Generator(nn.Module):
    def __init__(self,number_row):
        super(Generator,self).__init__()
        self.line1 = Linear1(number_row,number_row,bias = False)
       
       
    def forward(self,x,choice=None):
        if choice is not None:
            x = self.line1(x,choice)
        else:
            x = self.line1(x)
        
        return x
class Generator2(nn.Module):
    def __init__(self,number_col):
        super(Generator2,self).__init__()
        self.line1 = Linear1(number_col,number_col,bias = False)
       
       
    def forward(self,x,choice=None):
        x = x.permute(1,0)
        if choice is not None:
            x = self.line1(x,choice)
        else:
            x = self.line1(x)
        x = x.permute(1,0)
        return x
def W2(data,truedata,LAMBDA_A = 0.01,LAMBDA_B = 0.01,sigma = 0.1,PROTECT_NUMBER = 1, n_iter=400):           
    epochs = 1 
    UPDATE_ITERS_A = 5# to accelarate to convergence
    UPDATE_ITERS_B = 5
    TRANS_ITERS = n_iter
    epsilon = 1e-2
    print('N_iter= ', n_iter)
    statistic = np.zeros(TRANS_ITERS*5)
    statistic.shape = (TRANS_ITERS,5) 
    use_cuda =torch.cuda.is_available()
    #use_cuda = False
    if use_cuda:
        gpu = 0
    datacopy = data
    number_row = data.shape[0]
    number_col = data.shape[1]
    zero = torch.zeros(min(number_row,number_col))
    one = torch.ones(min(number_row,number_col))
    zero[0:PROTECT_NUMBER]=1
    protect = zero
    unprotect = one-zero
    if use_cuda:
        protect =protect.cuda(gpu)
        unprotect =unprotect.cuda(gpu)
    X_data = []
    index = []
    for i in range(number_col):
        p =data[:,i]
        X_data.append(p)
        index.append(i)
    trainset = train_data(X_data,index)
    train_loader = DataLoader(dataset=trainset,batch_size=number_col,shuffle=False)
    
    T,G=Generator(number_row),Generator2(number_col) 
    if use_cuda:
        T = T.cuda(gpu)
        G = G.cuda(gpu)
    optimizerT = optim.Adam(T.parameters(), lr=1e-3,betas =(0.5,0.9))
    #optimizerT = optim.RMSprop(T.parameters(), lr=1e-3,alpha = 0)
    #optimizerT = optim.SGD(T.parameters(), lr=1e-3,momentum = 0.5)
    optimizerG = optim.Adam(G.parameters(), lr=1e-3,betas =(0.5,0.9))
    #optimizerG = optim.RMSprop(G.parameters(), lr=1e-3,alpha = 0)
    #optimizerG = optim.SGD(T.parameters(), lr=1e-3,momentum = 0.5)
    if use_cuda:
        T.line1.weight.data = torch.eye(number_row).cuda()
        G.line1.weight.data = torch.eye(number_col).cuda()
    else:
        T.line1.weight.data = torch.eye(number_row)
        G.line1.weight.data = torch.eye(number_col)
    
    for iters in range(TRANS_ITERS):
        
        for i, (data,_) in enumerate(train_loader):
            break
        data = data.float()
        if use_cuda:
            data = data.cuda(gpu)
        data = torch.mm(G.line1.weight,torch.mm(data,T.line1.weight.t()))
        data = np.array(data.detach().cpu())
        S,U,V = svd.svd(data,PROTECT_NUMBER)
        principle = np.dot(U,np.dot(np.diag(S),V))
        
        residue = data - principle
        if use_cuda:
            principle = torch.from_numpy(principle).cuda()
        else:
            principle = torch.from_numpy(principle)
        pre_principle = principle.float()
        principle = torch.mm(torch.inverse(G.line1.weight),torch.mm(pre_principle,torch.inverse(T.line1.weight.t())))
        #delta_principle = torch.norm(pre_principle - principle) / torch.norm(pre_principle)
        #print("Iter: {} Delta={:.2%}".format(iters, delta_principle))
        if use_cuda:
            residue = torch.from_numpy(residue).cuda()
        else:
            residue = torch.from_numpy(residue)
        residue = residue.float()
        residue = torch.mm(torch.inverse(G.line1.weight),torch.mm(residue,torch.inverse(T.line1.weight.t())))
        residue = residue.detach()
        for epoch in range(epochs):    
            for iter_g in range(UPDATE_ITERS_A):
                T.zero_grad()
                
                fake = T(residue)
                fake = torch.mm(G.line1.weight.detach(),fake)
                covariance = torch.mm(fake.t(),fake)    
                ident = torch.eye(min(number_row,number_col))
                if use_cuda:
                    ident = ident.cuda()
                
                #covariance += 0.001*ident
                '''
                u,s,v = torch.svd(covariance)
                s[s<0] = 0
                s = torch.sqrt(s)
                cov_sqrtinv = torch.mm(torch.mm(u,torch.diag(1/s)),v.t())
                cov_sqrtinv = cov_sqrtinv.detach()
                Wasserstein_Distance1 = torch.trace(covariance + sigma*sigma*ident-2*sigma*torch.mm(cov_sqrtinv,covariance ))
                '''
                u,s,v = torch.svd(fake)
                s = torch.abs(s)
                #print('coefficeint %.2f')
                Wasserstein_Distance1 = 1.0/((number_col-1)*(number_row-1))*torch.trace(covariance)*torch.trace(covariance)
                #print('Step1 Wdist: %.2f' % Wasserstein_Distance1.float())
                Wasserstein_Distance1 -= 2.0*sigma*math.sqrt(1.0/((number_col-1)*(number_row-1)))*torch.sum(s)**2
                #print('Step2 Wdist: %.2f' % Wasserstein_Distance1)
                # print('cov', torch.trace(covariance))
                # print('s', torch.sum(s))
                #print("wasserstein distance left=%f"%Wasserstein_Distance)
                '''
                if (number_col > number_row)|(number_col == number_row):
                    A = torch.mm(T.line1.weight,T.line1.weight)
                else:
                    A = torch.mm(T.line1.weight[choice,:],T.line1.weight[:,choice])
                '''
                A = torch.mm(T.line1.weight.t(),T.line1.weight)
                penalty = LAMBDA_A*torch.norm(A,1)
                penalty.backward()
                gradient1 = T.line1.weight.grad.detach().cpu().numpy()
                if (iter_g == UPDATE_ITERS_A-1)&(epoch == epochs-1):
                    Wasserstein_Distance1.backward()
                else:
                    Wasserstein_Distance1.backward(retain_graph=True)
                #T.line1.weight.grad = 0.5*(T.line1.weight.grad + T.line1.weight.grad.t()) #stay symmetric
                gradient2 = T.line1.weight.grad.detach().cpu().numpy()-gradient1
                rate = LA.norm(gradient1)/(LA.norm(gradient2) + epsilon)
                optimizerT.step()
                
            
           
            for iter_g in range(UPDATE_ITERS_B):
                G.zero_grad()
                
                fake = G(residue)
                fake = torch.mm(fake,T.line1.weight.detach().t())
                  
                    
                    
                covariance = torch.mm(fake.t(),fake)
                ident = torch.eye(min(number_col,number_row))
                if use_cuda:
                    ident = ident.cuda()
                
                #covariance += 0.001*ident
                '''
                try:
                    u,s,v = torch.svd(covariance)
                except:
                    cov = np.array(covariance.detach())
                    np.save("D:\\covriance.npy",cov)
                    cov = np.load("D:\\covriance.npy")
                    covariance = torch.from_numpy(cov).cuda() 
                    covariance += 0.001*ident
                    u,s,v = torch.svd(covariance)
                    
                s[s<0] = 0
                s = torch.sqrt(s)
                cov_sqrtinv = torch.mm(torch.mm(u,torch.diag(1/s)),v.t())
                cov_sqrtinv = cov_sqrtinv.detach()
                
                Wasserstein_Distance2 = torch.trace(covariance + sigma*sigma*ident-2*sigma*torch.mm(cov_sqrtinv,covariance ))
                '''
                u,s,v = torch.svd(fake)
                s = torch.abs(s)
                Wasserstein_Distance2 = 1.0/((number_col-1)*(number_row-1))*torch.trace(covariance)*torch.trace(covariance)-2*sigma*math.sqrt(1.0/((number_col-1)*(number_row-1)))*torch.sum(s)*torch.sum(s)
                #print("wasserstein distance right=%f"%Wasserstein_Distance)
                '''
                if (number_col < number_row)|(number_col == number_row):
                    B = torch.mm(G.line1.weight,G.line1.weight)
                else:
                    B = torch.mm(G.line1.weight[choice,:],G.line1.weight[:,choice])
                '''
                B = torch.mm(G.line1.weight.t(),G.line1.weight)
                penalty = LAMBDA_B*torch.norm(B,1)
                penalty.backward()
                if (iter_g == UPDATE_ITERS_B-1)&(epoch ==epochs-1):
                    Wasserstein_Distance2.backward()
                else:
                    Wasserstein_Distance2.backward(retain_graph=True)
                #G.line1.weight.grad = 0.5*(G.line1.weight.grad + G.line1.weight.grad.t()) #stay symmetric
                optimizerG.step()
                    
                
            
                
    
        prins = principle.detach()
        prins = prins.cpu().numpy()
        
        statistic[iters,0] = np.linalg.norm(prins-truedata.T)
        statistic[iters,1] = np.sqrt(statistic[iters,0]*statistic[iters,0]/(number_row*number_col))
        statistic[iters,2] = 20*np.log10(datacopy.max()/statistic[iters,1])
        statistic[iters,3] = Wasserstein_Distance2
        statistic[iters,4] = rate
        if (iters+1)%10==0:
            print(statistic[iters,:])
            print(iters+1)       
    
    result = statistic[iters-2:iters+1,:]
    print(result)
    return principle.detach().cpu().numpy(),T.line1.weight.detach().cpu().numpy(),G.line1.weight.detach().cpu().numpy(),statistic

