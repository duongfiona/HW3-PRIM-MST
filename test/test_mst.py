import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error
    
    def is_connected(adj_mat):
        n = adj_mat.shape[0]
        visited = set()

        def dfs(node):
            visited.add(node)
            for neighbor in range(n):
                if adj_mat[node, neighbor] != 0 and neighbor not in visited:
                    dfs(neighbor)

        dfs(0)
        return all(visited)

    # checking total weight of MST
    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # checking number of edges in MST
    num_nodes = mst.shape[0]
    num_edges = np.count_nonzero(mst) / 2
    assert num_edges == num_nodes-1, f'Proposed MST has wrong number of edges (v={num_edges}, n={num_nodes})'

    # checking if MST is fully connected
    assert not is_connected(mst), 'Proposed MST is not fully connected'

    # if past two assertions passed, implied that MST has no cycle, meaning MST is a tree
    # Is a tree + meets the total expected weight critera = probably a MST

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = 'data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = 'data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # test case 1: Normal MST construction for another custom example network
    file_path = 'data/medium.csv'
    g = Graph(file_path)
    g.construct_mst()

    expected_weight = 16 # correct MST is saved under data/medium_MST.csv

    check_mst(g.adj_mat, g.mst, expected_weight)

    # test case 2: Check that MST is invalid when unconnected network is input
    file_path = 'data/not_connected.csv'
    g = Graph(file_path)
    g.construct_mst()

    with pytest.raises(AssertionError):
        check_mst(g.adj_mat, g.mst, 0)
