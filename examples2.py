# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
import igraph as ig
import os
os.chdir('/Users/ejg19/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD/M4R cont/Github/Partial_observations')
from utils.utils_surfaces import find_surfaces_adaptive
from utils.utils_mc import transition_matrix, unembed_MC, get_original_MC
from utils.utils_hdi_ifs import HDI_Discrete

# %%
datasets = ['Curvilinear-Sierpinski-data.npy','Henon-data.npy']

#define observation functions
def y_coordinate(x):
    return x[:,1]

def identity(x):
    return x

dataset_kwargs = {
    'Curvilinear-Sierpinski-data.npy': {
        "delay":3,
        "observation_function": y_coordinate,
        "delta":0.15,
        "theta":0.2 
    },
    'Henon-data.npy': {
        "delay":3,
        "observation_function": y_coordinate,
        "delta":0.02,
        "theta":0.1
    }
}

# choose dataset 0 = curvilinear sierpisnki, 1 = henon
key = 1
assert key in [0,1]
dataset = datasets[key]
data = np.load('Data/'+datasets[key])
delay, observation_function, delta, theta  = dataset_kwargs[dataset].values()

observed_data = observation_function(data) #apply observation function to dataset

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(observed_data[:-2],observed_data[1:-1],observed_data[2:],s=0.5,alpha=0.1,c='k')
plt.show()
# %%
#######################################
#       Separate delay maps
#######################################

embedded_data = np.column_stack([observed_data[i:len(observed_data)+1-(delay)+i] for i in range(delay)]) #delay embed data

surfaces = find_surfaces_adaptive(pd.DataFrame(embedded_data), delta_min = delta,theta_min=theta,theta_max=2*theta,min_s=1000, break_theta=15)


fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection='3d')

for i in set(surfaces['labels'].dropna()):
    surface = embedded_data[surfaces['labels']==i]
    ax.plot(surface[:,0],surface[:,1],surface[:,2],'.',alpha = 0.4,label = str(i));
ax.view_init(elev=20., azim=30)
plt.legend()
plt.show()

#%%
#######################################
#    Discover original Markov chain
#######################################

embedded_TM = transition_matrix(surfaces['labels'].dropna().to_numpy()) #get embedded Markov chain transition matrix
print(embedded_TM)
embedded_TM[embedded_TM<=0.1] = 0


unembedded_graph, labels = unembed_MC(ig.Graph.Weighted_Adjacency(embedded_TM),delay-1) #Unembed MC
original_TM = np.array(unembedded_graph.get_adjacency(attribute="weight"))


# post-process check of Markov chain

#remove any erroneous labels
for i in range(1,len(surfaces['labels'])-1):
     if isinstance(surfaces['labels'][i-1],int) & isinstance(surfaces['labels'][i+1],int):
        if isinstance(surfaces['labels'][i],int):
            if not ((labels[surfaces['labels'][i-1]-1][1:] == labels[surfaces['labels'][i]-1][:-1]).all() ):
                surfaces.loc[i,'labels'] = np.nan

original_MC = get_original_MC(surfaces,labels)
print(original_TM)

#%%
#######################################
#       fit HDI model
#######################################

N = [1000,8000] #length of training trajectory
Maps = original_TM.shape[0]
LRs = [0.05,0.05] #learning rate
Degree_poly = [3,2]
Epochs = [1000,1000]

X_train_tensor = torch.tensor(observed_data[100:], dtype=torch.double)


for _ in range(200):
    print(_)
    #initialise model with n_hv hidden varables, quadratic basis function and n maps, sparsity penalty of 0
    n_hv = 1
    model= HDI_Discrete(n_hv,Degree_poly[key],Maps,0)
    model.train()

    #  Convert numpy arrays to PyTorch tensors

    optimizer = optim.Adam(model.parameters(), lr=LRs[key])

    # Step 8: Training Loop (example)
    num_epochs = Epochs[key]
    
    for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            out = model.prop(X_train_tensor[:n_hv+1].flip(0),original_MC['labels'].to_list()[100:],N[key], X_train_tensor)
            loss = torch.square((out[:,0]-X_train_tensor[n_hv+1:N[key]])).mean()
            loss+= model.reg()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch%20==0:
                print(loss.item(), epoch)
            epoch_loss += loss.item() 
            if np.isnan(epoch_loss) == True:
                 break
    
    if epoch == num_epochs-1:
         break




#######################################
#  Plot sample trajectory of new IFS
#######################################

model.eval()
print(model.training)
N=10000 #sample trajectory
MC = []
for i in range(N):
     MC.append(np.random.randint(0,Maps))

out = model.prop(X_train_tensor[:2].flip(0),MC,N, X_train_tensor)
out = out.detach().numpy()

                                                                                                                        #%%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(out[:-2,0], out[1:-1,0], out[2:,0], s=2, c=MC[1:-3] , cmap = 'tab20')
ax.scatter(observed_data[:-2],observed_data[1:-1],observed_data[2:],s=0.5,alpha=0.1,c='k')
ax.set_xlabel('$x_{n-2}$')
ax.set_ylabel('$x_{n-1}$')
ax.set_zlabel('$x_n$')
ax.set_box_aspect(aspect=None, zoom=0.8)
ax.view_init(elev=20., azim=30)
ax.set_title('delay embeddings')


plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)

ax2.set_title('Inferred system')
ax2.scatter(out[10:,0],out[10:,1],s=2,c=MC[10:-2])
ax2.set_xlabel('$x$')
ax2.set_ylabel('$h$')

ax1.scatter(data[:,0],data[:,1],s=2,c='b')
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('True system')
plt.show()
# %%
