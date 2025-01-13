from torch_geometric.utils import homophily, dense_to_sparse, to_dense_adj
from torch_geometric.transforms import KNNGraph
from sklearn.neighbors import kneighbors_graph
import torch

def homophil_index(x, y, k=[10,20,50]):
    for i in k:
        A = kneighbors_graph(x, i, include_self=False).toarray()
        a = dense_to_sparse(torch.Tensor(A))
        h_edge = homophily(a[0], y, method='edge')
        h_node = homophily(a[0], y, method='node')
        print('K: {},  H1: {},  H2: {}'.format(i, h_edge, h_node))

