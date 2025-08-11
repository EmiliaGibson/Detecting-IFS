import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import lstsq
from scipy.linalg import orth
from scipy.linalg import null_space

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
    # if np.linalg.matrix_rank(M) < M.shape[1]:
    #     # print('not coplanar')
    #     return 'not coplanar'
    O = orth(M)
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1

    # print(lstsq(A.T, b,rcond=None)[0], null_space(O.T))
    return null_space(O.T)[:,0] #lstsq(A.T, b,rcond=None)[0]


def find_start(X):
    '''
    X : n x d numpy array
    '''
    N,d = np.shape(X)
    H,edges = np.histogramdd(X)
    min_density = min(np.max(H),2*d+1)
    indices_low_density = np.column_stack(np.where((H > min_density) & (H <= np.mean(H[H>0]))))
    regions_low_density = [np.array([edges[i][j[i]] for i in range(d)]) for j in indices_low_density]
    if len(regions_low_density)>2:
        nbrs = NearestNeighbors(n_neighbors=d, algorithm='auto').fit(X)
        _, indices = nbrs.kneighbors(regions_low_density)
        
        indices = indices.flatten()
        start = np.random.choice(np.arange(len(indices)))
    
        return indices[start]
    else:
        return np.random.choice(np.arange(N))

def find_surface_no_recursion(Q,delta = 0.05,theta = 0.1, progress_updates=True):
    '''
    Input must already embedded
    Q: pandas dataframe
    X the d - dimensional surface
    delta
    theta
    '''
    n,d = Q.shape
    start = find_start(np.array(Q))
    X=np.array(Q)
    NN = Q.iloc[start]
    S = [np.concatenate((NN.values,[NN.name]))]
    E = []
    # Q = pd.DataFrame(X)
    normals = []
    nbrs = NearestNeighbors(n_neighbors=d, algorithm='ball_tree').fit(X) #n_neighbours = dimension
    distances, indices = nbrs.kneighbors([X[start,:]])
    while len(S) + len(E) < n:
        # print(len(S))
        # if len(S)>1000:
        #     break
        if progress_updates == True:
            progressbar(len(S) + len(E),n)
        if len(normals)>0: 
            #SPEED UP ALGORITHM 
            #You only need to another NN search when you append to S

            #check at least one point is within distance delta to assigned points
            if np.min(distances)>delta:
                break
            #new point being examined
            min_row = np.argmin(distances[:,0], axis=None)

            NN = Q.iloc[min_row]

            #nearest neighbours in S
            # print(indices[min_row])
            S_NN = [np.array(S)[:,:-1][indices[min_row,i]] for i in list(range(d-1))]

            #normal at i1
            normal = normals[indices[min_row,0]-1]

            # #get second nearest neighbour in S
            # i2 = S[indices[min_row,1]]

            new_normal = find_orth(S_NN - NN.values)
            # if isinstance(new_normal, str):
            #     S_NN = [S[indices[min_row,i]] for i in list(range(d))]
            #     new_normal = find_orth(S_NN - X[NN,:])
            #     print(new_normal)
            Q = Q.drop(NN.name)

            # print(angle_between(normal,new_normal),NN.name)

            if angle_between(normal,new_normal)<theta:
                normals.append(new_normal)
                S.append(np.concatenate((NN.values,[NN.name])))

                nbrs = NearestNeighbors(n_neighbors=d, algorithm='ball_tree').fit(np.array(S)[:,:-1])
                distances, indices = nbrs.kneighbors(Q.to_numpy())
         
            else:

                E.append(np.concatenate((NN.values,[NN.name])))
                distances = np.delete(distances, (min_row), axis=0)
                indices = np.delete(indices, (min_row), axis=0)

            
        else:
            NNs = indices[0]
            points = [np.concatenate((Q.iloc[j].values,[Q.iloc[j].name])) for j in NNs]
            # i1 = indices[0][1]
            # i2 = indices[0][2]
            S.extend(points[1:])
            points = [X[j,:] for j in NNs]
            normal = find_orth(points[:-1]-points[-1])
            # if isinstance(normal, str):
            #     print('intialisation is coplanar')
            #     break
            # np.cross((X[i1,:] - X[i2,:])/np.linalg.norm(X[i1,:] -X[i2,:]),
            #                   (X[i,:] - X[i2,:])/np.linalg.norm(X[i,:] -X[i2,:]))
            NNs = Q.iloc[indices[0]]
            Q = Q.drop(NNs.index)
            normals += [normal for _ in list(range(d-1))]
            nbrs = NearestNeighbors(n_neighbors=d, algorithm='ball_tree').fit(np.array(S)[:,:-1])
            distances, indices = nbrs.kneighbors(Q.to_numpy())

    
    if len(E)>0:
        E = np.array(E)

        add_df = pd.DataFrame(E[:,:-1],index=E[:,-1].astype(np.int64), columns=Q.columns)

        Q = pd.concat([Q, add_df]).sort_index()


    S = np.stack(S,axis = 0)
    return  S, Q



def find_surfaces_adaptive(Q,delta_min = 0.05,delta_max = 0.15,theta_min = 0.001,theta_max=0.3,break_theta=10,break_delta=3,min_s=100):
    out = pd.DataFrame(index=Q.index, columns=['labels'])
    N,_ = Q.shape
    theta = theta_min
    delta=delta_min
    num_surfaces=1
    break1=0
    break2=0
    text = lambda theta, delta, N, num_surfaces : f"""
        {'-'*40}
        # Current Values
        # theta: {theta}
        # delta: {delta}
        # N: {N}
        # Surfaces: {num_surfaces-1}

        {'-'*40}
        """

    print(text(theta,delta,N,num_surfaces))
    while N > min_s:
        S,Q_new = find_surface_no_recursion(Q,delta=delta,theta=theta, progress_updates=True)

        print('len =', len(S))

        if len(S) > min_s:
            out.loc[S[:,-1]]= num_surfaces
            num_surfaces+=1
            Q= Q_new
            N = len(Q_new)
        else: 
            break1+=1
        
        if break1 > break_theta:
            break2+=1

            if break2<break_delta:
                theta+=theta_min
                break1=0
                print(text(theta,delta,N,num_surfaces))
            else:
                delta+=delta_min
                break2=0
                break1=0
                print(text(theta,delta,N,num_surfaces))
        if theta > theta_max:
            break
        if delta>delta_max:
            break
    
    return out



