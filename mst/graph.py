import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        V = list(range(self.adj_mat.shape[0]))
    
        s = 0 # arbitrarily initializing to start at 0th node
        S = set([s])

        # keeping dict of known cheapest costs from S to node "key" 
        lowest_costs = {v: float('inf') for v in V}
        lowest_costs[s] = 0

        # keeping track of edges used to get to node "key" from S
        pred = {v: None for v in V}
        pred[s] = s

        # initializing pq
        pq = []
        for v in V:
            heapq.heappush(pq, (lowest_costs[v], v))
        
        while pq:
            _, u = heapq.heappop(pq)
            S.add(u)

            u_neighbors = self.adj_mat[u].nonzero()[0]
            for v in u_neighbors:
                if v not in S:
                    cost_to_v = self.adj_mat[u, v]
                    if cost_to_v < lowest_costs[v]:
                        lowest_costs[v] = cost_to_v
                        pred[v] = u
                        heapq.heappush(pq, (cost_to_v, v))


        # assembling MST using info from pred and lowest_costs
        self.mst = np.zeros(self.adj_mat.shape)
        for u, v in pred.items():
            self.mst[u, v] = lowest_costs[u]
            self.mst[v, u] = lowest_costs[u]
        
        return lowest_costs
