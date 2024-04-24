import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import lstsq
from scipy.linalg import orth

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    theta = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return min(theta,np.pi - theta)

def progressbar(i,n,refresh_rate = 1, i_start = 0):
    '''
    Approximates how far through the computation you are, by showing a percentage of total loops that have been computed
    i = loop index~~~~~~~~~~~~~~~~
    n = amount of iterations (max loop index + 1 if i_start = 0)
    refresh_rate = how often you would like to update the percentage (useful)
    i_start = start value of loop
    '''
    output = (i-i_start)/refresh_rate
    if (int(output)==output):
        print(str(np.round((i+1-i_start)/n*100,2))+'%', end ="\r")
        
def find_orth(O):
    M = np.stack(O,axis = -1)
    # get 5 orthogonal vectors in 10 dimensions in a matrix form
    O = orth(M)
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b,rcond=None)[0]

#function implementing the algorithm outlined above
def find_curves_2D(X,eps = 0.05,eps_grad = 0.1):
    '''
    X the (n-1) - dimensional surface
    eps
    eps_grad
    '''
    n = len(X)
    start = np.random.choice(np.arange(n))
    A1 = [X[start,:]]
    A2 = []
    df = pd.DataFrame(X)
    normals = []
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X) #n_neighbours = dimension
    distances, indices = nbrs.kneighbors([X[start,:]])
    while len(A1) + len(A2) < n:
        progressbar(len(A1) + len(A2),n)
        if len(normals)>0: 
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(A1)
            distances, indices = nbrs.kneighbors(df.to_numpy())
            #check at least one point is within distance epsilon to assigned points
            if np.min(distances)>eps:
                break
            #new point being examined
            min_row = np.argmin(distances[:,0], axis=None)

            NN = df.iloc[min_row].name
            df = df.drop(NN)
            #nearest neighbour in A1
            i1 = A1[indices[min_row,0]]

            #normal at i1
            normal = normals[indices[min_row,0]-1]

            #get second nearest neighbour in A1
            i2 = A1[indices[min_row,1]]

            new_normal = np.cross((i1- X[NN,:])/
                                    np.linalg.norm(i1 -X[NN,:]),
                                  (i2 - X[NN,:])/np.linalg.norm(i2 -X[NN,:]))
            if angle_between(normal,new_normal)<eps_grad:
                normals.append(new_normal)
                A1.append(X[NN,:])
         
            else:
                A2.append(X[NN,:])
                    
            
        else:
            i = start
            i1 = indices[0][1]
            i2 = indices[0][2]
            A1.extend([X[i2,:],X[i1,:]])
            normal = np.cross((X[i1,:] - X[i2,:])/np.linalg.norm(X[i1,:] -X[i2,:]),
                              (X[i,:] - X[i2,:])/np.linalg.norm(X[i,:] -X[i2,:]))
            df = df.drop([i,i1,i2])

            normals.append(normal)
            normal = np.cross((X[i2,:] - X[i1,:])/np.linalg.norm(X[i2,:] -X[i1,:]),
                              (X[i,:] - X[i1,:])/np.linalg.norm(X[i,:] -X[i1,:]))
            normals.append(normal)
            i = i2
    

    if len(A2)>0:
        A2 = np.stack(A2,axis = 0)
        A2 = np.vstack((A2,df.to_numpy()))
    else:
        A2 = df.to_numpy()
    A1 = np.stack(A1,axis = 0)
    #recursively continue identify distinct curves from leftover unassigned points
    if len(A2)<10:
        if len(A1)<10:
            return []
        else:
            return [A1] 
    elif len(A1)<10:
        return [] + find_curves_2D(A2,eps,eps_grad)
    else: 
        return [A1] + find_curves_2D(A2,eps,eps_grad)

  def find_curves_3D(X,eps = 0.05,eps_grad = 0.1):
    '''
    X the (n-1) - dimensional surface
    eps
    eps_grad
    '''
    n = len(X)
    #randomly select an initial point
    start = np.random.choice(np.arange(n))
    #assign this point to the surface we are exploring
    A1 = [X[start,:]]
    
    A2 = [] #set of explored points
    df = pd.DataFrame(X) #set of unexplored points

    normals = [] #set of approximate normals to the surface

    #find the 3 nearest neighbours to our surface 
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X) #n_neighbours = dimension
    distances, indices = nbrs.kneighbors([X[start,:]])

    #begin nearest neighbour search
    while len(A1) + len(A2) < n:
        #progress tracker
        progressbar(len(A1) + len(A2),n)

        if len(normals)>0: 
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(A1)
            distances, indices = nbrs.kneighbors(df.to_numpy())
            #check at least one point is within distance epsilon to assigned points
            if np.min(distances)>eps:
                break
            #new point being examined
            min_row = np.argmin(distances[:,0], axis=None)
            NN = df.iloc[min_row].name
            #remove point from set of unexplored points
            df = df.drop(NN)

            #nearest neighbour in A1
            i1 = A1[indices[min_row,0]]
            #normal at i1
            normal = normals[indices[min_row,0]]
            #get second nearest neighbour in A1
            i2 = A1[indices[min_row,1]]
            #get third nearest neighbour in A1
            i3 = A1[indices[min_row,2]]

            new_normal = find_orth([i1-X[NN,:], i2-X[NN,:], i3-X[NN,:]])
            if angle_between_3d(normal,new_normal)<eps_grad:
                #assign to surface and store normal
                normals.append(new_normal)
                A1.append(X[NN,:])
            else:
                #assign to set of explored points
                A2.append(X[NN,:])
            
            
        # initialise surface and normal to surface    
        else:
            i = start
            i1 = indices[0][1]
            i2 = indices[0][2]
            i3 = indices[0][3]
            #assign nearest neighbours to the surface
            A1.extend([X[i2,:],X[i1,:],X[i3,:]])
            #approximate normal vector
            normal = find_orth([X[i1,:] - X[i2,:],X[i,:] - X[i2,:],X[i3,:] - X[i2,:]])     
            #remove points from unexplored set
            df = df.drop([i,i1,i2,i3])
            #store the approximated normals at each point
            normals += [normal,normal,normal,normal]

    #empty the set of unexplored points and covert to numpy array
    if len(A2)>0:
        A2 = np.stack(A2,axis = 0)
        A2 = np.vstack((A2,df.to_numpy()))
    else:
        A2 = df.to_numpy()
    #convert list to numpy array of dimension 3
    A1 = np.stack(A1,axis = 0)
    #recursively continue identify distinct curves from leftover unassigned points
    if len(A2)<20000:
        if len(A1)<50:
            return []
        else:
            return [A1] 
    elif len(A1)<50:
        return [] + find_curves_3D(A2,eps,eps_grad)
    else: 
        return [A1] + find_curves_3D(A2,eps,eps_grad)

