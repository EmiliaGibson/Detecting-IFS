import numpy as np
import matplotlib.pyplot as plt
from utils import find_surfaces


datasets = ['Curvilinear-Sierpinski-data.npy','Henon-data.npy','Sierpinski-data.npy','Logistic-data.npy']

#dictionary with algorithms parameters for each dataset
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

# choose dataset 0 = curvilinear sierpisnki, 1 = henon, 2 = sierpinski, 3 = logistic  
key = 2
assert key in [0,1,2,3]
dataset = datasets[key]

#load dataset
data = np.load('Data/' + datasets[key])

#embed the data to higher dimension
embedded_data = np.column_stack((data[:-1,:],data[1:,0]))

#apply separation algorithm
surfaces = find_surfaces(embedded_data, **dataset_kwargs[dataset])


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

