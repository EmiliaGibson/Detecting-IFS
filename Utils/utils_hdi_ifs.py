#lets start by believing the world is polynomial
#%%
# model 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import comb
from itertools import product



#-------------------------------------------------------------------
#                    Step 1: Define the Network
#-------------------------------------------------------------------
class HDI_Discrete(nn.Module):

    def __init__(self, M, d, k, reg_coef):
        '''
        Initialise model parameters of HDI framework

        Parameters
        ---------------
        M : int 
            Number of hidden varaibles to initialise
        d : int
            Max degree of class of polynomials to consider
        k : int
            number of maps in ifs
        reg_coef: float
            sparsity penalties in polynomial regression

        Returns
        ---------------
        poly_coeffs: np.ndarray
            array containing polynomial coeeficients for each hidden variable
        powers: np.ndarray
            1d array containing the powers of each variable appearing for each 
            element in a row of  poly_coeffs
        '''
        super(HDI_Discrete, self).__init__()
        poly_coeffs = np.random.uniform(-1,1,size=(k,M+1,comb(M+1+d,d)))
        poly_coeffs[:,:,0] = 1
        self.poly_coeffs = torch.nn.Parameter(torch.tensor(0.5*poly_coeffs,dtype=torch.double))
        powers = np.array(tuple([ x for x in product(range(d+1),repeat = M+1) if sum(x)<= d]) )
        powers = powers[np.argsort(powers.sum(axis=1))]
        self.powers = torch.tensor(powers, dtype=torch.int32)
        self.reg_coef = reg_coef
        self.M = M
        self.k = k

    def forward(self,inputs,omega):
        '''
        Parameters
        -------------------
        inputs: 1 dimensional tensor of length M+1
        omega: int
            which map to apply

        '''
        return (torch.pow(inputs,self.powers).prod(axis=1)*self.poly_coeffs[omega]).sum(axis=1)
    
    def reg(self):
        data_loss = 0
        for omega in range(self.k):
            for j in range(self.poly_coeffs.shape[1]):
                    # sp = np.sign(poly_coeffs[:,j])
                    sqn = torch.sqrt(1+self.powers[j].sum())
                    data_loss += (self.reg_coef*sqn *torch.abs(self.poly_coeffs[omega,:,j])).sum()
        return data_loss
    
    def prop(self,initial_condition,jump_chain,N,data):
        '''
        propogate model forward under current parameters

        Parameters
        -------------------
        input: zero-dimensional tensor

        N: int
            length of trajectory

        Returns
        -------------------
        outputs: 1 dimensional tensor of length N-1
        
        '''
        outputs = torch.zeros(N-self.M-1,self.M+1)
        input = initial_condition
        for i in range(N-self.M-1):
            if isinstance(jump_chain[i],int):
                input = self.forward(input,jump_chain[i])
                outputs[i] = input
            else:
                inputs = torch.zeros(self.k,self.M+1)
                for j in range(self.k):
                    inputs[j] = self.forward(input,j)
                input = inputs[torch.argmin(torch.abs(inputs[:,0]- data[self.M+1+i]))]
                outputs[i] = input
            if self.training == True:
                if i % 5 == 0:
                # if torch.abs(input[0])>torch.max(torch.abs(data)):
                    input[0] = data[self.M+1+i]
        return outputs



#%%


# def MSE(outputs, targets):
#     return np.mean(np.square(outputs-targets[1:]))

# def regularization(poly_coeffs, powers, reg_coef):
#     data_loss = 0
#     for j in range(poly_coeffs.shape[1]):
#             # sp = np.sign(poly_coeffs[:,j])
#             sqn = np.sqrt(1+powers[j].sum())
#             data_loss += (reg_coef*sqn *np.abs(poly_coeffs[:,j])).sum()
#     return data_loss


# # input = np.array([targets[0]] + list(np.ones(M)))

# def prop(input, poly_coeffs, powers,N):
#     outputs = np.zeros(N-1)
#     for i in range(N-1):
#         input = iterate_forward(input, poly_coeffs, powers)
#         outputs[i] = input[0]

#     return outputs





         

    

# %%
