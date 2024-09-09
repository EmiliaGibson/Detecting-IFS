import numpy as np
import matplotlib.pyplot as plt
from utils import find_surfaces

datasets = ['Curvilinear-Sierpinski-data.npy','Henon-data.npy','Sierpinski-data.npy','Logistic-data.npy']


dataset_kwargs = {
    'Curvilinear-Sierpinski-data.npy': {
        "delta":0.15,
        "theta":1.75
    },
    'Henon-data.npy': {
        "delta":0.15,
        "theta":0.15
    },
    'Sierpinski-data.npy': {
        "delta":0.1,
        "theta":1.75
    },
    'Logistic-data.npy': {
        "delta":0.05,
        "theta":0.1
    }
}

# choose dataset 0 = curvilinear sierpisnki, 1 = henon or 2 = sierpinski 
key = 1
assert key in [0,1,2,3]
dataset = datasets[key]
data = np.load('Data/'+datasets[key])

embedded_data = np.column_stack((data[:-1,:],data[1:,:]))
surfaces = find_surfaces(embedded_data, **dataset_kwargs[dataset])


#plot the different surfaces and their respective densities
if key in [0,1,2]:
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    for i in range(len(surfaces)):
        print(len(surfaces[i]))
        if len(surfaces[i])>0:
            ax.plot(surfaces[i][:,0],surfaces[i][:,1],surfaces[i][:,2],'.',alpha = 0.4,label = str(i));
    plt.show()

elif key == 3:
    fig = plt.figure(figsize=(6,6))
    for i in range(len(surfaces)):
        print(len(surfaces[i]))
        if len(surfaces[i])>0:
            plt.plot(surfaces[i][:,0],surfaces[i][:,1],'.',alpha = 0.4,label = str(i));


distr = []
for i in range(len(surfaces)):
    if len(surfaces[i])>1:
        distr.append(len(surfaces[i]))

fig = plt.figure(figsize=(6,6))
plt.bar(np.arange(1,len(surfaces)+1),distr/np.sum(distr))
plt.ylabel('density')
plt.xlabel('$f_i$')

