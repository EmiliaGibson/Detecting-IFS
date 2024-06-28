#%%
import numpy as np
import matplotlib.pyplot as plt
from utils import find_surfaces

#%%
datasets = ['Curvilinear-Sierpinski-data.npy','Henon-data.npy','Sierpinski-data.npy','Logistic-data.npy']

#
dataset_kwargs = {
    'Curvilinear-Sierpinski-data.npy': {
        "delta":0.15,
        "theta":0.3
    },
    'Henon-data.npy': {
        "delta":0.1,
        "theta":0.25
    },
    'Sierpinski-data.npy': {
        "delta":0.1,
        "theta":0.25
    },
    'Logistic-data.npy': {
        "delta":0.05,
        "theta":0.1
    }
}

# choose dataset 0 = curvilinear sierpisnki, 1 = henon or 2 = sierpinski 
key = 2
assert key in [0,1,2,3]
dataset = datasets[key]
data = np.load(datasets[key])

embedded_data = np.column_stack((data[:-1,:],data[1:,0]))
# a = dataset_kwargs['Henon-data.npy'][key1]
# b = dataset_kwargs['Henon-data.npy'][key2]
surfaces = find_surfaces(embedded_data, **dataset_kwargs[dataset])

#%%
#plot the different surfaces and their respective densities
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')
for i in range(len(surfaces)):
    print(len(surfaces[i]))
    if len(surfaces[i])>0:
        ax.plot(surfaces[i][:,0],surfaces[i][:,1],surfaces[i][:,2],'.',alpha = 0.4,label = str(i));
plt.show()


distr = []
for i in range(len(surfaces)):
    if len(surfaces[i])>1:
        distr.append(len(surfaces[i]))

plt.bar(np.arange(1,len(surfaces)+1),distr/np.sum(distr))
plt.ylabel('density')
plt.xlabel('$f_i$')
#%%
def g(x,j):
    if j == 0:
        return 4*x*(1-x)
    if j == 1:
        return 3.8*x*(1-x)
    if j == 2:
        return 3.6*x*(1-x)
    else:
        return ValueError('j != 0,1,2')
    
#%%
np.random.seed(10)
N = 10000
A = np.zeros((N,1))
A[0,:] = np.random.uniform(0,1)
colours = [-1]
for i in range(N-1):
    j = np.random.randint(0,3)
    colours.append(j)
    A[i+1, :] = g(A[i,:],j)
np.save('Logistic-data.npy',A)
# %%
