import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
import igraph as ig
from Utils.utils_surfaces import find_surfaces_adaptive
from Utils.utils_mc import transition_matrix, unembed_MC, get_original_MC
from Utils.utils_hdi_ifs import HDI_Discrete


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
        "delta":0.1,
        "theta":0.05
    }
}

# choose dataset 0 = curvilinear sierpisnki, 1 = henon
key = 1
assert key in [0,1]
dataset = datasets[key]
data = np.load('Data/'+datasets[key])
delay, observation_function, delta, theta  = dataset_kwargs[dataset].values()

observed_data = observation_function(data) #apply observation function to dataset

embedded_data = np.column_stack([observed_data[i:len(observed_data)+1-(delay)+i] for i in range(delay)]) #delay embed data

surfaces = find_surfaces_adaptive(pd.DataFrame(embedded_data), delta_min = delta,theta_min=theta,theta_max=3*theta,min_s=1000)



embedded_TM = transition_matrix(surfaces['labels'].dropna().to_numpy()) #get embedded Markov chain transition matrix
embedded_TM[embedded_TM<=0.06] = 0


unembedded_graph, labels = unembed_MC(ig.Graph.Weighted_Adjacency(embedded_TM),delay-1) #Unembed MC
original_TM = np.array(unembedded_graph.get_adjacency(attribute="weight"))


# post-process check of Markov chain

#remove any erroneous labels
for i in range(1,len(surfaces['labels'])-1):
     if isinstance(surfaces['labels'][i-1],int) & isinstance(surfaces['labels'][i+1],int):
        if isinstance(surfaces['labels'][i],int):
            if not ((labels[surfaces['labels'][i-1]-1][1:] == labels[surfaces['labels'][i]-1][:-1]).all() ):
                print(labels[surfaces['labels'][i-1]-1],labels[surfaces['labels'][i]-1],labels[surfaces['labels'][i+1]-1] )
                print('fail')
                surfaces.loc[i,'labels'] = np.nan

original_MC = get_original_MC(surfaces,labels)


#fit HDI model
N = [4000,5000]
Maps = [3,2]
LRs = [0.05,0.01]
Degree_poly = [3,2]
Epochs = 2000

X_train_tensor = torch.tensor(observed_data[100:], dtype=torch.double)
for _ in range(1000):
    print(_)
    #initialise model with n_hv hidden varables, quadratic basis function and n maps, sparsity penalty of 0
    n_hv = 1
    model= HDI_Discrete(n_hv,Degree_poly[key],Maps[key],0)
    model.train()

    #  Convert numpy arrays to PyTorch tensors

    optimizer = optim.Adam(model.parameters(), lr=LRs[key])

    # Step 8: Training Loop (example)
    num_epochs = Epochs
    
    for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            out = model.prop(X_train_tensor[:n_hv+1].flip(0),original_MC['labels'].to_list()[100:],N[key], X_train_tensor)
            loss = torch.square((out[:,0]-X_train_tensor[n_hv+1:N[key]])).mean()
            loss+= model.reg()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch%20==0:
                print(loss.item())
            epoch_loss += loss.item() 
            if np.isnan(epoch_loss) == True:
                 break
    
    if epoch == num_epochs-1:
         break



N=10000 #sample trajectory
MC = []
for i in range(N):
     MC.append(np.random.randint(0,Maps[key]))

out = model.prop(X_train_tensor[:2].flip(0),MC,N, X_train_tensor[:2].flip(0))
out = out.detach().numpy()


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
