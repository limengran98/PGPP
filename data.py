import os
import os.path as osp
from argparse import Namespace
from collections import defaultdict
from typing import Callable, Optional

import pickle as pkl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import datasets
from torch_sparse import SparseTensor
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected

from torch_geometric.data import Data
import torch_geometric.utils
from scipy import io

def load_heter_data(dataset_name):

    DATAPATH = 'data/heterophily_datasets_matlab'
    fulldata = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')

    edge_index = fulldata['edge_index'] 
    node_feat = fulldata['node_feat']   
    label = np.array(fulldata['label'], dtype=np.int32).flatten()  
    num_features = node_feat.shape[1]
    num_classes = np.max(label) + 1  
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(node_feat, dtype=torch.float)  
    y = torch.tensor(label, dtype=torch.long) 
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    data = Data(x=x, edge_index=edge_index, y=y)

    return data, num_features, num_classes

class HeterophilousGraphDataset(InMemoryDataset):
    r"""The heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Roman-empire"`,
            :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"`,
            :obj:`"Questions"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Roman-empire
          - 22,662
          - 32,927
          - 300
          - 18
        * - Amazon-ratings
          - 24,492
          - 93,050
          - 300
          - 5
        * - Minesweeper
          - 10,000
          - 39,402
          - 7
          - 2
        * - Tolokers
          - 11,758
          - 519,000
          - 10
          - 2
        * - Questions
          - 48,921
          - 153,540
          - 301
          - 2
    """
    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace('-', '_')
        assert self.name in [
            'roman_empire',
            'amazon_ratings',
            'minesweeper',
            'tolokers',
            'questions',
        ]

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self):
        raw = np.load(self.raw_paths[0], 'r')
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'
    



def is_large(data):
    """
    Return whether a dataset is large or not.
    """
    return data == 'arxiv'


def is_continuous(data):
    """
    Return whether a dataset has continuous features or not.
    """
    return data in ['pubmed', 'coauthor', 'arxiv']


def to_edge_tensor(edge_index):
    """
    Convert an edge index tensor to a SparseTensor.
    """
    row, col = edge_index
    value = torch.ones(edge_index.size(1))
    return SparseTensor(row=row, col=col, value=value)


def validate_edges(edges):
    """
    Validate the edges of a graph with various criteria.
    """
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError()

    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError()
        m[src].add(dst)

    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError()





def load_arxiv(root):
    """
    Load the Arxiv dataset, which is not included in PyG.
    """
    features = torch.from_numpy(np.load(f'{root}/ArXiv/x.npy'))
    labels = torch.from_numpy(np.load(f'{root}/ArXiv/y.npy'))
    edge_index = torch.from_numpy(np.load(f'{root}/ArXiv/edge_index.npy'))
    return Namespace(data=Namespace(x=features, y=labels, edge_index=edge_index))


def load_data(args, split=None, seed=None, normalize=False,
              validate=False):
    """
    Load a dataset from its name.
    """
    root = '../data'
    dataset = args.data
    seed = 0

    if dataset in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
        data = HeterophilousGraphDataset(root, name=dataset)
    elif dataset in ['squirrel', 'chameleon','genius']:
        DATAPATH = "/home/lmr/TAMG/data/"
        fulldata = io.loadmat(f'{DATAPATH}/{dataset}.mat')
        edge_index = fulldata['edge_index'] 
        node_feat = fulldata['node_feat']  
        label = np.array(fulldata['label'], dtype=np.int32).flatten()  

        num_features = node_feat.shape[1]
        num_classes = np.max(label) + 1 
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(node_feat, dtype=torch.float)  
        y = torch.tensor(label, dtype=torch.long)  

        edge_index = torch_geometric.utils.to_undirected(edge_index)
        edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
        data = Data(x=x, edge_index=edge_index, y=y)
    else:
        raise ValueError(dataset)
    if dataset in ['squirrel', 'chameleon','genius']:
        node_x = data.x
        node_y = data.y
        edges = data.edge_index
    else:
        node_x = data.data.x
        node_y = data.data.y
        edges = data.data.edge_index

    if validate:
        validate_edges(edges)

    if normalize:
        assert (node_x < 0).sum() == 0  # all positive features
        norm_x = node_x.clone()
        norm_x[norm_x.sum(dim=1) == 0] = 1
        norm_x = norm_x / norm_x.sum(dim=1, keepdim=True)
        node_x = norm_x

    if split is None:
        if hasattr(data.data, 'train_mask'):
            trn_mask = data.data.train_mask
            val_mask = data.data.val_mask
            trn_nodes = torch.nonzero(trn_mask).view(-1)
            val_nodes = torch.nonzero(val_mask).view(-1)
            test_nodes = torch.nonzero(~(trn_mask | val_mask)).view(-1)
        else:
            trn_nodes, val_nodes, test_nodes = None, None, None
    elif len(split) == 2:
        trn_size, test_size = split
        indices = np.arange(node_x.shape[0])
        trn_nodes, test_nodes = train_test_split(indices, test_size=test_size, random_state=seed,
                                                 stratify=node_y)

        trn_nodes = torch.from_numpy(trn_nodes)
        test_nodes = torch.from_numpy(test_nodes)

    if dataset in ['squirrel', 'chameleon','genius']:
        return data, trn_nodes, test_nodes
    else:
        return data[0], trn_nodes, test_nodes

def main():
    """
    Main function.
    """
    for data in ['cora', 'citeseer', 'photo', 'computers', 'steam', 'pubmed',
                 'coauthor']:
        load_data(data, split=(0.4, 0.1, 0.5), validate=True, verbose=True)


if __name__ == '__main__':
    main()
