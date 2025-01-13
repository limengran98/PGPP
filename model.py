from collections import defaultdict
from itertools import permutations
import random
import torch
from torch import Tensor
from torch import nn
from torch_scatter import scatter_add
import torch_geometric
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, remove_self_loops

def get_propagation_matrix(edge_index: Adj, n_nodes: int, mode: str = "adj") -> torch.sparse.FloatTensor:
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=n_nodes, normalization="sym")
    if mode == "adj":
        edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj

def get_edge_index_from_y(y: torch.Tensor, know_mask: torch.Tensor = None) -> Adj:
    nodes = defaultdict(list)
    label_idx_iter = enumerate(y.numpy()) if know_mask is None else zip(know_mask.numpy(),y[know_mask].numpy())
    for idx, label in label_idx_iter:
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T




def get_edge_index_from_y_ratio(y: torch.Tensor, ratio: float = 1.0) -> torch.Tensor:
    n = y.size(0)
    mask = []
    nodes = defaultdict(list)
    for idx, label in random.sample(list(enumerate(y.numpy())), int(ratio*n)):
        mask.append(idx)
        nodes[label].append(idx)
    arr = []
    for v in nodes.values():
        arr += list(permutations(v, 2))
    return torch.tensor(arr, dtype=torch.long).T, torch.tensor(mask, dtype=torch.long)


def to_dirichlet_loss(attrs, laplacian):
    return torch.bmm(attrs.t().unsqueeze(1), laplacian.matmul(attrs).t().unsqueeze(2)).view(-1).sum()

class arbLabel:

    def __init__(self, edge_index: Adj, x: torch.Tensor, y: torch.Tensor, know_mask: torch.Tensor, is_binary = False):
        self.x = x
        self.y = y
        self.n_nodes = x.size(0)
        self.edge_index = edge_index
        self._adj = None

        self._label_adj = None
        self._label_adj_25 = None
        self._label_adj_50 = None
        self._label_adj_75 = None
        self._label_adj_all = None
        self._label_mask = know_mask
        self._label_mask_25 = None
        self._label_mask_50 = None
        self._label_mask_75 = None

        self.know_mask = know_mask
        self.mean = 0 if is_binary else self.x[self.know_mask].mean(dim=0)
        self.std = 1 #if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (self.x[self.know_mask]-self.mean) / self.std
        # init self.out without normalized
        self.out = torch.zeros_like(self.x)
        self.out[self.know_mask] = self.x[self.know_mask]

    @property
    def adj(self):
        if self._adj is None:
            self._adj = get_propagation_matrix(self.edge_index, self.n_nodes)
        return self._adj

    def label_adj(self):
        if self._label_adj is None:
            edge_index = get_edge_index_from_y(self.y, self.know_mask)
            self._label_adj = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj, self._label_mask
    
    def label_adj_25(self):
        if self._label_adj_25 is None:
            _, label_mask_50 = self.label_adj_50()
            self._label_mask_25 = torch.tensor(random.sample(label_mask_50.tolist(), int(0.5*label_mask_50.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_25)
            self._label_adj_25 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_25, self._label_mask_25

    def label_adj_50(self):
        if self._label_adj_50 is None:
            _, label_mask_75 = self.label_adj_75()
            self._label_mask_50 = torch.tensor(random.sample(label_mask_75.tolist(), int(0.75*label_mask_75.size(0))),dtype=torch.long)
            edge_index = get_edge_index_from_y(self.y, self._label_mask_50)
            self._label_adj_50 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_50, self._label_mask_50

    def label_adj_75(self):
        if self._label_adj_75 is None:
            edge_index, self._label_mask_75 = get_edge_index_from_y_ratio(self.y, 0.75)
            self._label_adj_75 = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_75, self._label_mask_75

    @property
    def label_adj_all(self):
        if self._label_adj_all is None:
            edge_index = get_edge_index_from_y(self.y)
            self._label_adj_all = get_propagation_matrix(edge_index, self.n_nodes)
        return self._label_adj_all

    def arb(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, num_iter: int = 1, **kw) -> torch.Tensor:
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def _arb_label(self, adj: Adj, mask:torch.Tensor, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1):
        G = torch.ones(self.n_nodes)
        G[mask] = gamma
        G = G.unsqueeze(1)
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = G*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-G)*torch.spmm(adj, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean

    def arb_label_25(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_25()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_50(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_50()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_75(self, out: torch.Tensor = None, alpha: float = 0.85, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        adj,mask=self.label_adj_75()
        return self._arb_label(adj,mask,out,alpha,beta,gamma,num_iter)

    def arb_label_100(self, out: torch.Tensor = None, alpha: float = 0.95, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*(alpha*torch.spmm(self.adj, out)+(1-alpha)*out.mean(dim=0)) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean
    
    def arb_label_all(self, out: torch.Tensor = None, beta: float = 0.70, gamma:float = 0.75, num_iter: int = 1, **kw):
        if out is None:
            out = self.out
        out = (out - self.mean) / self.std
        for _ in range(num_iter):
            out = gamma*torch.spmm(self.adj, out) + (1-gamma)*torch.spmm(self.label_adj_all, out)
            out[self.know_mask] = beta*out[self.know_mask] + (1-beta)*self.out_k_init
        return out * self.std + self.mean


class arbLoss(nn.Module):
    def __init__(self, edge_index: Adj, raw_x: torch.Tensor, know_mask: torch.Tensor, alpha, beta, device, is_binary=False, **kw):
        super().__init__()

        self.device = device
        num_nodes = raw_x.size(0)
        self.n_nodes = num_nodes
        num_attrs = raw_x.size(1)
        self.know_mask = know_mask.to(device)  # Ensure mask is on the same device

        self.mean = 0 if is_binary else raw_x[know_mask].mean(dim=0).to(device)
        self.std = 1  # if is_binary else x[know_mask].std(dim=0)
        self.out_k_init = (raw_x[know_mask] - self.mean) / self.std

        edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        self.L = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(num_nodes, num_nodes)).to_dense().to(device)
        self.avg_L = num_nodes / (num_nodes - 1) * torch.eye(num_nodes, device=device) - 1 / (num_nodes - 1) * torch.ones(num_nodes, num_nodes, device=device)
        
        self.x = nn.Parameter(torch.zeros(num_nodes, num_attrs, device=device))
        self.x.data[know_mask] = raw_x[know_mask].clone().detach().data.to(device)
        
        # Default values for alpha and beta
        if alpha == 0:
            alpha = 0.00001
        if beta == 0:
            beta = 0.00001
        self.theta = (1 - 1 / num_nodes) * (1 / alpha - 1)
        self.eta = (1 / beta - 1) / alpha
        # print(alpha, beta, self.theta, self.eta)

    def get_loss(self, x):
        x = (x - self.mean) / self.std
        dirichlet_loss = to_dirichlet_loss(x, self.L)
        avg_loss = to_dirichlet_loss(x, self.avg_L)
        recon_loss = nn.functional.mse_loss(x[self.know_mask], self.out_k_init, reduction="sum")
        return -(dirichlet_loss + self.eta * recon_loss + self.theta * avg_loss)

    def forward(self):
        return self.get_loss(self.x)

    def get_out(self):
        return self.x