import pandas as pd
import json
import pickle as pkl
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch


class GraphEmo(Data):

    # Create a graph for
        # single person
        # single movie
        # single timepoint

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, adj = None, movie=None, subject=None, timestap_sec=None):
        
        # In case a specific adj is passed, used it for the connectivity
        if adj != None:
            # Create edge_index tensor
            edge_index = torch.tensor(adj.nonzero().t().contiguous(), dtype=torch.long)  # Shape: [2, num_edges]
            print(edge_index)
            print(edge_index.shape)
            # Create edge attributes: weights can be set as 1 or extracted from the adj_matrix if applicable
            edge_attr = torch.tensor(adj[edge_index[0, :], edge_index[1, :]], dtype=torch.float)
            print(edge_attr)
            print(edge_attr.shape)
        
        # build adj from adge attr and index 
        if adj == None:
            adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)
   
        super(GraphEmo, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Add attr
        self.movie = movie
        self.subject = subject
        self.timestamp_sec = timestap_sec
        self.adj = adj
