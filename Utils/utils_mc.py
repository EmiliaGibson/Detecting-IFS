#%%
import numpy as np
import networkx as nx
from utils.markovchain1 import MarkovChain
from collections import deque
import pandas as pd
import igraph as ig



# %%
def transition_matrix(data):
    '''
    code taken from https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
    '''
    n = int(max(data)) #number of states assuming states are labelled 1,..,n

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(data,data[1:]):
        M[i-1][j-1] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return np.array(M)






#%%
def get_embedded_MC(mat,delay):
    #state space
    E= list(range(mat.shape[0]))
    N=100000

    data=[]
    data.append(int(np.random.choice(E)))
    embedded_data = []
    for i in range(N-1):
        data.append(int(np.random.choice(E,p=mat[data[-1]])))
        if i+1 > delay-1:
            embedded_data.append(tuple(data[-delay:]))
        # convert embedded chain from tuples to unique integers
    delay_chain = np.zeros(N-delay,dtype=np.int32)
    for j in range(len(set(embedded_data))):
        delay_chain[np.all(np.array(embedded_data) ==np.array(list(set(embedded_data))[j]), axis=1)] = j

    return np.array(transition_matrix(delay_chain))


# %%


# havent used a directed cycle basis


def unembed_MC(Q,m):
    '''
    Q: Network of embedded MC
    return P: square numpy array (original probability matrix)

    '''
    P = ig.Graph(directed=True)
    
    

    #adjacency matric
    Q_adj = np.array(Q.get_adjacency(attribute = "weight"))
    elementary_cycles = list(nx.recursive_simple_cycles(nx.DiGraph(Q_adj)))
    elementary_cycles.sort(key=len)

    labels = np.array([ [None for __ in range(m)] for _ in range(Q_adj.shape[0])])
    

    #create first closed loop in P from Q
    loop = elementary_cycles.pop(0)
    n = len(loop)
    P.add_vertex(str(loop[0]))
    for i in range(n):
        prob = Q_adj[loop[i%n],loop[(i+1)%n]]
        if i != n-1:
            P.add_vertex(str(loop[i+1]))
        P.add_edge(str(loop[i%n]),str(loop[(i+1)%n]),weight = prob)
        labels[loop[i]] = [loop[(i - j)%n] for j in range(m-1,-1,-1)] 
    
    seen_nodes = loop

    while (labels==None).any() and len(elementary_cycles)>0:
        P, elementary_cycles, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,elementary_cycles,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)
    # P, cycle_basis, seen_nodes,labels = examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels)

    #remove multi-edges and combine weights by taking average
    P.simplify(loops = False, combine_edges="mean")
    return  P, labels







def examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels):
    loop = cycle_basis.pop(0)
    print('loop = ',loop)
    for node in loop:

        #shortest path from new node to seen_nodes, d(node,L)
        path_from = min(Q.get_shortest_paths(node,seen_nodes, output="vpath",mode='out'),key=len) 
        print(path_from)
        if len(path_from)-1 < m :
            labels = get_shared_labels(m,len(path_from)-1,labels,path_from[0],path_from[-1],direction ='out')
            print(labels[loop])
            #prop2.5ii
            labels = rotate_loop_labels(m,loop,node,labels)
            print(labels[loop])

        #shortest path from seen nodes to new node, d(L,node)
        path_to = min(Q.get_shortest_paths(node,seen_nodes, output="vpath",mode='in'),key=len)
        print(path_to)
        if len(path_to) -1 < m :
            labels = get_shared_labels(m,len(path_to)-1,labels,path_to[0],path_to[-1], direction = 'in')
            print(labels[loop])
            labels = rotate_loop_labels(m,loop,node,labels)
            print(labels[loop])


    if (labels[loop]==None).any() == False:
        #update adjacency matrix
        n = len(loop)
        for i in range(n):
            prob = Q_adj[loop[i%n],loop[(i+1)%n]]
            P.add_edge(str(labels[loop[i%n],-1]),str(labels[loop[(i+1)%n],-1]),weight = prob)


        return P, cycle_basis, seen_nodes,labels

    else:
        #add new nodes to P as needed
        while (labels[loop]==None).any():
            indices = np.where(labels[loop]==None)
            #get new label that has not been used already!!
            idx = 0
            label =  loop[indices[0][0]]
            while label in seen_nodes:
                idx+=1
                label = loop[indices[0][idx]]
            
            node = loop[indices[0][idx]]
            labels[node][indices[1][idx]] = label
            P.add_vertex(str(label))
            seen_nodes.append(label)
            deque_loop = deque(loop)
            for j in range(1,int(m)):
                if (indices[1][idx]-j) >= 0:
                    #shift loop by sigma
                    deque_loop.rotate(-1)
                    node = deque_loop[indices[0][idx]]
                    labels[node][(indices[1][idx]-j)] =  label
            
            deque_loop = deque(loop)
            for j in range(1,int(m)):
                if (indices[1][idx]+j) < m :
                    #shift loop by inverse sigma
                    deque_loop.rotate(1)
                    node = deque_loop[indices[0][idx]]
                    labels[node][(indices[1][idx]+j)] =  label
        
        #add probabilities to P 
        loop = list(loop)
        n=len(loop)
        for i in range(n):
            prob = Q_adj[loop[i%n],loop[(i+1)%n]]
            P.add_edge(str(labels[loop[i%n],-1]),str(labels[loop[(i+1)%n],-1]),weight = prob)


        return P, cycle_basis, seen_nodes,labels


def rotate_loop_labels(m,loop,node,labels):
    #apply prop 2.5 ii
    deque_loop = deque(loop)
    idx = loop.index(node)
    label = labels[node]
    for j in range(1,int(m)):
        #shift loop by sigma
        deque_loop.rotate(-1)
        #shift labels by sigma
        node = deque_loop[idx]

        #final elements are unknown
                #only update not None elements
        labels[node][:-j][label[j:]!= None] =  label[j:][label[j:]!= None]
    
    deque_loop = deque(loop)
    for j in range(1,int(m)):
        #shift loop by inverse sigma
        deque_loop.rotate(1)
        #shift labels by inverse sigma
        node = deque_loop[idx]
        #first elements are unknown
            #only update not None elements
        labels[node][j:][label[:-j]!= None] = label[:-j][label[:-j]!= None]
    return labels


def get_shared_labels(m,dist, labels, unlabelled_node, labelled_node, direction):
    if direction == 'out':
        print(labelled_node,labels[labelled_node])

        #From new node to seen_nodes, d(node,L)
        labels[unlabelled_node][dist:] = labels[labelled_node][:m-dist]

    if direction =='in':
        print(labelled_node,labels[labelled_node])
        #From seen nodes to new node, d(L,node)
        labels[unlabelled_node][:m-dist] = labels[labelled_node][dist:]

    return labels






# def transition_matrix(data):
#     '''
#     code adapted from https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python
#     '''
#     n = int(max(data)) #number of states assuming states are labelled 1,..,n

#     M = [[0]*n for _ in range(n)]

#     for (i,j) in zip(data,data[1:]):
#         if isinstance(i,int) and isinstance(j,int):
#             M[i-1][j-1] += 1

#     #now convert to probabilities:
#     for row in M:
#         s = sum(row)
#         if s > 0:
#             row[:] = [f/s for f in row]
#     return np.array(M)

# def get_shared_nodes(m,Q,dists,paths,nearest_loops,loop,labels):
#     loop = deque(loop)
#     for i in range(len(dists)):
#         nearest_loop = deque(nearest_loops[i])
#         idx = nearest_loop.index(paths[i][0])
#         idx2 = loop.index(paths[i][-1])

#         if dists[i]>0:
#             loop.rotate(int(dists[i]))
#             labels[loop[idx2]] = labels[paths[i][0]]
#             for j in range(int(m-dists[i]-1)):
#                 nearest_loop.rotate(1)
#                 loop.rotate(1)
#                 labels[loop[idx2]] = labels[nearest_loop[idx]]
#         if dists[i]==0:
#             intersecting_nodes = list(set(nearest_loop).intersection(loop))
#             # for each node in the new loop that is distance one (behind) shared node in common
#             # add the m labels shared in the original MC
#             out = np.array([nx.multi_source_dijkstra(Q,set(nearest_loop).difference(loop),j,weight=None) for j in intersecting_nodes],dtype=object)
#             for j in np.where(out[:,0]==1)[0]:
#                 node = intersecting_nodes[j]
#                 idx = nearest_loop.index(node)
#                 idx2 = loop.index(node)
#                 labels[loop[idx2]] = labels[nearest_loop[idx]]
#                 for j in range(m-1):
#                     nearest_loop.rotate(1)
#                     loop.rotate(1)
#                     labels[loop[idx2]] = labels[nearest_loop[idx]]

#     return labels


# def create_new_loop(Q_adj,P,loop,labels):
#     n = len(loop)
#     for x in loop:
#         if labels[x] == None:
#             labels[x] = x
#     for i in range(n):
#         prob = Q_adj[loop[i%n],loop[(i+1)%n]]
#         P.add_edge(labels[loop[i%n]],labels[loop[(i+1)%n]],weight = prob)

#     return P, labels
    
# def connect_smallest_closed_loops(Q,Q_adj,m,P,closed_loops,seen_loops,labels):
#     #now find nearest loop to found loops
#     while len(closed_loops)>0:
#         #if adjacency matrix is a probability matrix we are done!
#         if np.linalg.norm(np.sum(nx.adjacency_matrix(P).todense(),axis=-1)-np.ones(len(P.nodes)))<0.1:      
#             return P, closed_loops, seen_loops, labels
#         for x in closed_loops:
#             dists,paths,nearest_loops = np.inf*np.ones(len(seen_loops)),np.zeros(len(seen_loops),dtype=object),np.zeros((len(seen_loops)),dtype=object)
#             for j in x:
#                 out = np.array([nx.multi_source_dijkstra(Q,set(l),j,weight=None) for l in seen_loops],dtype=object)
#                 indices = np.where(dists-out[:,0]>0)[0]
#                 if len(indices)>0:
#                     dists[indices] = out[indices,0]
#                     paths[indices] = out[indices,1]
#                     nearest_loops[indices] = np.array(seen_loops,dtype=object)[indices].tolist()
#             # nearest_loops = nearest_loops.reshape(paths.shape)
            
#             # original MC is connected so there will always exist a loop which attaches to the starting loop
#             if np.min(dists)<m:

#                 #keep cycles that are of a distance < m only
#                 indices = np.where(dists<m)[0]
#                 dists = dists[indices]
#                 paths = paths[indices]
#                 nearest_loops = nearest_loops[indices]

#                 loop = x
#                 closed_loops.remove(loop)

#                 #check to see loop corresponds to a simple closed loop in P
#                 if len([j for j in range(len(dists)) if dists[j] < m - len(nearest_loops[j])])>0:
#                     break
#                 # if dist_to_nearest_loop < m - len(nearest_loop):
#                 #     break
#                 subsets = [l for l in seen_loops if set(l).issubset(loop)==True  ]
#                 if len(subsets)>0:
#                     break    

#                 #so loop corresponds to simple closed loop in P
#                 seen_loops.append(loop)
#                 # get labels of nodes in common
#                 labels =  get_shared_nodes(m,Q,dists,paths,nearest_loops,loop,labels)
#                 #create new loop
#                 P,labels = create_new_loop(Q_adj,P,loop,labels)

#                 #find next nearest and shortest cycle
#                 break
                
#         if np.min(dists) >= m:
#             break
#     return P, closed_loops, seen_loops, labels

# def unembed_MC(Q,m):
#     '''
#     Q: Network of embedded MC
#     return P: square numpy array (original probability matrix)

#     '''
#     P = nx.DiGraph()
#     closed_loops = list(nx.simple_cycles(Q))
#     closed_loops.sort(key=len)
#     #adjacency matric
#     Q_adj = nx.adjacency_matrix(Q).todense()
#     labels = [None for _ in list(range(Q_adj.shape[0]))]
    

#     #create first closed loop in P from Q
#     loop = closed_loops.pop(0)
#     n = len(loop)
#     for i in range(n):
#         prob = Q_adj[loop[i%n],loop[(i+1)%n]]
#         P.add_edge(loop[i%n],loop[(i+1)%n],weight = prob)
#         labels[loop[i]] =loop[i]
    
#     seen_loops = [loop]
#     P, closed_loops, seen_loops,labels = connect_smallest_closed_loops(Q,Q_adj,m,P,closed_loops,seen_loops,labels)
         

#     return  P, labels

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


def get_original_MC(embedded_MC,labels):
    ''''
    embedded_MC: pandas dataframe
    '''
    original_MC = pd.DataFrame(index=embedded_MC.index, columns=['labels'])
    sorted_labels = list(set(labels[:,-1]))
    for i in embedded_MC['labels'].unique():
        if isinstance(i,int):
            original_MC.loc[embedded_MC['labels']==i] = sorted_labels.index(labels[i-1,-1])
    return original_MC
# %%
