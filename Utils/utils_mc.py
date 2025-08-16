#%%
import numpy as np
import networkx as nx
from collections import deque
import pandas as pd
import igraph as ig




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

    #remove multi-edges and combine weights by taking average
    P.simplify(loops = False, combine_edges="mean")
    return  P, labels







def examine_next_loop(Q,Q_adj,m,P,cycle_basis,seen_nodes,labels):
    loop = cycle_basis.pop(0)
    for node in loop:

        #shortest path from new node to seen_nodes, d(node,L)
        path_from = min(Q.get_shortest_paths(node,seen_nodes, output="vpath",mode='out'),key=len) 
        if len(path_from)-1 < m :
            labels = get_shared_labels(m,len(path_from)-1,labels,path_from[0],path_from[-1],direction ='out')
            #prop2.5ii
            labels = rotate_loop_labels(m,loop,node,labels)

        #shortest path from seen nodes to new node, d(L,node)
        path_to = min(Q.get_shortest_paths(node,seen_nodes, output="vpath",mode='in'),key=len)
        if len(path_to) -1 < m :
            labels = get_shared_labels(m,len(path_to)-1,labels,path_to[0],path_to[-1], direction = 'in')
            labels = rotate_loop_labels(m,loop,node,labels)


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

        #From new node to seen_nodes, d(node,L)
        labels[unlabelled_node][dist:] = labels[labelled_node][:m-dist]

    if direction =='in':
        #From seen nodes to new node, d(L,node)
        labels[unlabelled_node][:m-dist] = labels[labelled_node][dist:]

    return labels




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

