import numpy as np
import matplotlib.pyplot as plt

#functions in IFS
def g(x):
    return 1+x[1] - 1.2*x[0]**2,0.3 * x[0]
def h(x):
    return 1 + x[1] - 1.2*(x[0] - 0.2)**2, -0.2 * x[0]


#generate data set
np.random.seed(10)
N = 10000
A = np.zeros((N,2))
A[0,:] = np.random.uniform(0,1,2)
for i in range(N-1):
    j = np.random.randint(0,2)
    if j ==0:
        A[i+1, :] = g(A[i,:])
    elif j ==1:
        A[i+1,:] = h(A[i,:])
attractor = A

#identify first coordinate maps
lift1 = np.stack((attractor[:-1,0],attractor[:-1,1],attractor[1:,0]),axis = -1)
test = find_curves_2D(lift1,eps=0.1,eps_grad=0.25)

#identify second coordinate maps
lift2 = np.stack((attractor[:-1,0],attractor[:-1,1],attractor[1:,1]),axis = -1)
test2 = find_curves_2D(lift2,eps=0.15,eps_grad=0.1)
