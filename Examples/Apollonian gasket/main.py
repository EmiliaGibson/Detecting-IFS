import numpy as np
import matplotlib.pyplot as plt

def f(z):
    '''
      Equations of IFS
    '''
    f =  ((np.sqrt(3) - 1)*z+1)/(-z + np.sqrt(3)+1)

    return f, f*np.exp(2*np.pi*1j/3), f*np.exp(-2*np.pi*1j/3)


#generate dataset
N = 10000
A = np.zeros(N,dtype=np.complex_)
A[0] = 1+0j
for i in range(N-1):
    j = np.random.randint(0,3)
    A[i+1] = f(A[i])[j]

#move data from complex plane to 2d plane
attractor = A
attractor = np.stack((attractor.real,attractor.imag),axis = -1)

#lift to graoh of first coordinate map
lift = np.stack((attractor[:-1,0],attractor[:-1,1],attractor[1:,0]),axis = -1)

#apply algorithm to identify different surfaces
test = find_curves_2D(B2,eps=0.15,eps_grad=0.3)
