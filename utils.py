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
    '''
    find vector orthogonal to a set of vectors
    '''
    M = np.stack(O,axis = -1)
    O = orth(M)
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b,rcond=None)[0]

def find_surfaces(X,delta = 0.05,theta = 0.1, progress_updates=True):
    '''
    Input must already embedded
    X the d - dimensional surface
    delta
    theta
    '''
    n,d = np.shape(X)
    start = np.random.choice(np.arange(n))
    S = [X[start,:]]
    E = []
    Q = pd.DataFrame(X)
    normals = []
    nbrs = NearestNeighbors(n_neighbors=d, algorithm='ball_tree').fit(X) #n_neighbours = dimension
    distances, indices = nbrs.kneighbors([X[start,:]])
    while len(S) + len(E) < n:
        if progress_updates == True:
            progressbar(len(S) + len(E),n)
        if len(normals)>0: 
            nbrs = NearestNeighbors(n_neighbors=d-1, algorithm='ball_tree').fit(S)
            distances, indices = nbrs.kneighbors(Q.to_numpy())
            #check at least one point is within distance delta to assigned points
            if np.min(distances)>delta:
                break
            #new point being examined
            min_row = np.argmin(distances[:,0], axis=None)

            NN = Q.iloc[min_row].name
            Q = Q.drop(NN)

            #nearest neighbours in S
            # print(indices[min_row])
            S_NN = [S[indices[min_row,i]] for i in list(range(d-1))]

            #normal at i1
            normal = normals[indices[min_row,0]-1]

            # #get second nearest neighbour in S
            # i2 = S[indices[min_row,1]]

            new_normal = find_orth(S_NN - X[NN,:])

            if angle_between(normal,new_normal)<theta:
                normals.append(new_normal)
                S.append(X[NN,:])
         
            else:
                E.append(X[NN,:])
            
            
        else:
            NNs = indices[0]
            points = [X[j,:] for j in NNs]
            # i1 = indices[0][1]
            # i2 = indices[0][2]
            S.extend(points[1:])
            normal = find_orth(points[:-1]-points[-1])
            # np.cross((X[i1,:] - X[i2,:])/np.linalg.norm(X[i1,:] -X[i2,:]),
            #                   (X[i,:] - X[i2,:])/np.linalg.norm(X[i,:] -X[i2,:]))
            Q = Q.drop(NNs)
            normals += [normal for j in list(range(d-1))]
            i = points[-1]
    
    if len(E)>0:
        E = np.stack(E,axis = 0)
        E = np.vstack((E,Q.to_numpy()))
    else:
        E = Q.to_numpy()
    S = np.stack(S,axis = 0)
    #recursively continue identify distinct curves from leftover unassigned points
    # stop recursion when <100 points remain in E
    if len(E)<100:
        #discard S if S only has <100 points assigned to it
        if len(S)<100:
            return []
        else:
            return [S] 
    #discard S if S only has <100 points assigned to it
    elif len(S)<100:
        return [] + find_surfaces(E,delta,theta)
    else: 
        return [S] + find_surfaces(E,delta,theta)