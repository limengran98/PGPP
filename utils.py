import argparse

from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops
import numpy as np
from greatx.functional import spmm
import torch_geometric
from torch_scatter import scatter_add
from torch_geometric.typing import Adj, OptTensor
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, remove_self_loops
from torch_geometric.nn import GCNConv, GATConv, SGConv, TransformerConv, SAGEConv, DirGNNConv,GPSConv
def test_val(feature, p=0.84):
    rows = feature.shape[0]
    num_rows = int(rows * p)
    sampled = feature[:num_rows] 
    return sampled

def relabel_nodes(index, edge_index):
    index_map = {old: new for new, old in enumerate(index)}
    print(index_map)
    edge_index = [index_map[i] for i in edge_index]
    return edge_index

def sample_y_nodes(num_nodes, y, y_ratio, seed):
    """
    Sample nodes with observed labels.
    """
    if y_ratio == 0:
        y_nodes = None
        y_labels = None
    elif y_ratio == 1:
        y_nodes = torch.arange(num_nodes)
        y_labels = y[y_nodes]
    else:
        y_nodes, _ = train_test_split(np.arange(num_nodes), train_size=y_ratio, random_state=seed,
                                      stratify=y.numpy())
        y_nodes = torch.from_numpy(y_nodes)
        y_labels = y[y_nodes]
    return y_nodes, y_labels


@torch.no_grad()
def to_f1_score(input, target, epsilon=1e-8):
    """
    Compute the F1 score from a prediction.
    """
    assert (target < 0).int().sum() == 0
    tp = ((input > 0) & (target > 0)).sum()
    fp = ((input > 0) & (target == 0)).sum()
    fn = ((input <= 0) & (target > 0)).sum()
    return (tp / (tp + (fp + fn) / 2 + epsilon)).item()


@torch.no_grad()
def to_recall(input, target, k):
    """
    Compute the recall score from a prediction.
    """
    pred = input.topk(k, dim=1, sorted=False)[1]
    row_index = torch.arange(target.size(0))
    target_list = []
    for i in range(k):
        target_list.append(target[row_index, pred[:, i]])
    num_pred = torch.stack(target_list, dim=1).sum(dim=1)
    num_true = target.sum(dim=1)
    return (num_pred[num_true > 0] / num_true[num_true > 0]).mean().item()


@torch.no_grad()
def to_ndcg(input, target, k, version='sat'):
    """
    Compute the NDCG score from a prediction.
    """
    if version == 'base':
        return ndcg_score(target, input, k=k)
    elif version == 'sat':
        device = target.device
        target_sorted = torch.sort(target, dim=1, descending=True)[0]
        pred_index = torch.topk(input, k, sorted=True)[1]
        row_index = torch.arange(target.size(0))
        dcg = torch.zeros(target.size(0), device=device)
        for i in range(k):
            dcg += target[row_index, pred_index[:, i]] / np.log2(i + 2)
        idcg_divider = torch.log2(torch.arange(target.size(1), dtype=float, device=device) + 2)
        idcg = (target_sorted / idcg_divider).sum(dim=1)
        return (dcg[idcg > 0] / idcg[idcg > 0]).mean().item()
    else:
        raise ValueError(version)


@torch.no_grad()
def to_r2(input, target):
    """
    Compute the CORR (or the R square) score from a prediction.
    """
    a = ((input - target) ** 2).sum()
    b = ((target - target.mean(dim=0)) ** 2).sum()
    return (1 - a / b).item()


@torch.no_grad()
def to_rmse(input, target):
    """
    Compute the RMSE score from a prediction.
    """
    return ((input - target) ** 2).mean(dim=1).sqrt().mean().item()


def print_log(epoch, loss_list, acc_list):
    """
    Print a log during the training.
    """
    print(f'{epoch:5d}', end=' ')
    print(' '.join(f'{e:.4f}' for e in loss_list), end=' ')
    print(' '.join(f'{e:.4f}' for e in acc_list))

def compute_f_n2d(edge_index, mask, out):
    nv = out.shape[0]
    len_v_0tod_list = []
    f_n2d = torch.zeros(nv, dtype = torch.int)
    v_0 = torch.nonzero(out[:,0]).view(-1)
    len_v_0tod_list.append(len(v_0))
    v_0_to_now = v_0
    f_n2d[v_0] = 0
    d = 1
    while True:
        v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0]
        v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
        if len(v_d) == 0:
            break
        f_n2d[v_d] = d
        v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
        len_v_0tod_list.append(len(v_d))
        d += 1
    return f_n2d


def get_normalized_adjacency(edge_index, n_nodes, mode=None):
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    if mode == "left":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight
    elif mode == "right":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = edge_weight * deg_inv_sqrt[col]
    elif mode == "article_rank":
        d = deg.mean()
        deg_inv_sqrt = (deg+d).pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight
    else:
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD


def get_propagation_matrix(edge_index, n_nodes, mode="adj"):
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    # edge_index, edge_weight = get_normalized_adjacency(edge_index, n_nodes)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj


class FeaturePropagation(nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: torch.Tensor, edge_index: Adj, mask: torch.Tensor) -> torch.Tensor:
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]
        else:
            out = x.clone()

        n_nodes = x.shape[0]
        adj = get_propagation_matrix(edge_index, n_nodes)
        for _ in range(self.num_iterations):
            out = torch.spmm(adj, out)
            out[mask] = x[mask]
        return out

class APA:
    def __init__(self, edge_index: Adj, x: torch.Tensor, know_mask: torch.Tensor, is_binary=True):
        self.edge_index = edge_index
        self.x = x
        self.n_nodes = x.size(0)
        self.know_mask = know_mask
        self.mean = 0 if is_binary else x[know_mask].mean(dim=0)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask] - self.mean) / self.std
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]
        self._adj = None

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def fp(self, out: torch.Tensor = None, num_iter: int = 30, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = self.out_k_init
        # f_n2d = compute_f_n2d(self.edge_index, self.know_mask, out).repeat(self.out.shape[1],1)
        # cor = torch.corrcoef(out.T).nan_to_num().fill_diagonal_(0)
        # f_n2d = f_n2d.to(out.device)
        # a_1 = (0.95 ** f_n2d.T) * (out - torch.mean(out, dim=0))
        # a_2 = torch.matmul(a_1, cor)
        # out_1 = 1 * (1 - (0.95 ** f_n2d.T)) * a_2
        # out = out + out_1
        return out #* self.std + self.mean

    def fp_analytical_solution(self, **kw) -> torch.Tensor:
        adj = self.adj.to_dense()

        assert self.know_mask.dtype == torch.int64
        know_mask = torch.zeros(self.n_nodes, dtype=torch.bool)
        know_mask[self.know_mask] = True
        unknow_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        unknow_mask[self.know_mask] = False

        A_uu = adj[unknow_mask][:, unknow_mask]
        A_uk = adj[unknow_mask][:, know_mask]

        L_uu = torch.eye(unknow_mask.sum()) - A_uu
        L_inv = torch.linalg.inv(L_uu)

        out = self.out.clone()
        out[unknow_mask] = torch.mm(torch.mm(L_inv, A_uk), self.out_k_init)

        return out * self.std + self.mean

    def pr(self, out: torch.Tensor = None, alpha: float = 0.55, num_iter: int = 30, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def ppr(self, out: torch.Tensor = None, alpha: float = 0.85, weight: torch.Tensor = None, num_iter: int = 50,
            **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        if weight is None:
            weight = self.mean
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * weight
            out[self.know_mask] = self.out_k_init
        return out * self.std + self.mean

    def mtp(self, out: torch.Tensor = None, beta: float = 0.85, num_iter: int = 50, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = torch.spmm(self.adj, out)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def mtp_analytical_solution(self, beta: float = 0.85, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        eta = (1 / beta - 1)
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik), eta * torch.mm(Ik, out))
        return out * self.std + self.mean

    def umtp(self, out: torch.Tensor = None, alpha: float = 0.99, beta: float = 0.9, num_iter: int = 2,
             **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp2(self, out: torch.Tensor = None, alpha: float = 0.9, beta: float = 0.2, gamma: float = 0.75,
              num_iter: int = 20, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma * (alpha * torch.spmm(self.adj, out) + (1 - alpha) * out.mean(dim=0)) + (1 - gamma) * out
            out[self.know_mask] = beta * out[self.know_mask] + (1 - beta) * self.out_k_init
        return out * self.std + self.mean

    def umtp_analytical_solution(self, alpha: float = 0.85, beta: float = 0.70, **kw) -> torch.Tensor:
        n_nodes = self.n_nodes
        theta = (1 - 1 / self.n_nodes) * (1 / alpha - 1)
        eta = (1 / beta - 1) / alpha
        edge_index, edge_weight = get_laplacian(self.edge_index, num_nodes=n_nodes, normalization="sym")
        L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to_dense()
        Ik_diag = torch.zeros(n_nodes)
        Ik_diag[self.know_mask] = 1
        Ik = torch.diag(Ik_diag)
        L1 = torch.eye(n_nodes) * (n_nodes / (n_nodes - 1)) - torch.ones(n_nodes, n_nodes) / (n_nodes - 1)
        out = (self.out - self.mean) / self.std
        out = torch.mm(torch.inverse(L + eta * Ik + theta * L1), eta * torch.mm(Ik, out))
        return out * self.std + self.mean

def to_device(gpu):
    """
    Return a PyTorch device from a GPU index.
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def str2bool(v):
    """
    Convert a string variable into a bool.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False
    else:
        raise argparse.ArgumentTypeError()


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.conv1 = GATConv(in_channels, hidden_channels, heads = 8)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads = 1)
        #GINConv(GCNConv(in_channels, hidden_channels))
        #GINConv(GCNConv(hidden_channels, out_channels))
    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x =  F.log_softmax(x, dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        return x

class Linear(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Linear, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
    def forward(self, x):
        x = self.lin1(x)#.relu()
        x = self.lin2(x)
        return x#F.log_softmax(x, dim=1)


from torch_geometric.nn import GCNConv, GATConv
# 定义不同的模型架构
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.ae = nn.Linear(output_dim, input_dim)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        re_x = self.ae(x)
        return x, re_x
    def re(self, x):
        re_x = self.ae(x)
        return re_x


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class AP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, output_dim, args):
        super(AP, self).__init__()
        if args.MPNN =='Dir':
            self.conv1 = DirGNNConv(TransformerConv(num_features, hidden_channels))
            self.conv2 = DirGNNConv(TransformerConv(hidden_channels, output_dim))
        # if args.MPNN =='GAT':
        #     self.conv1 = DirGNNConv(GATConv(num_features, hidden_channels))
        #     self.conv2 = DirGNNConv(GATConv(hidden_channels, hidden_channels))
        # if args.MPNN =='GraphSAGE':
        #     self.conv1 = DirGNNConv(SAGEConv(num_features, hidden_channels))
        #     self.conv2 = DirGNNConv(SAGEConv(hidden_channels, hidden_channels))
        if args.MPNN =='MLP':
            self.conv1 = torch.nn.Linear(num_features, hidden_channels)
            self.conv2 = torch.nn.Linear(hidden_channels, output_dim)

        if args.MPNN =='GCN':
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, output_dim)
            
        if args.MPNN =='GAT':
            self.conv1 = GATConv(num_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, output_dim)
        if args.MPNN =='GraphSAGE':
            self.conv1 = SAGEConv(num_features, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, output_dim)
        if args.MPNN =='Transformer':
            self.conv1 = TransformerConv(num_features, hidden_channels)
            self.conv2 = TransformerConv(hidden_channels, output_dim)
        if args.MPNN =='GATv2':
            self.conv1 = GATv2Conv(num_features, hidden_channels)
            self.conv2 = GATv2Conv(hidden_channels, output_dims)
        if args.MPNN =='SGC':
            self.conv1 = SGConv(num_features, hidden_channels)
            self.conv2 = SGConv(hidden_channels, output_dim)
        if args.MPNN =='Cluster':
            self.conv1 = ClusterGCNConv(num_features, hidden_channels)
            self.conv2 = ClusterGCNConv(hidden_channels, output_dim)
        if args.MPNN =='GPS':
            self.conv1 = GPSConv(num_features, GCNConv(num_features, num_features), attn_type = 'performer')
            self.conv2 = GCNConv(num_features, output_dim)
        self.dropout = args.dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x