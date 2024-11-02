import pandas as pd
import json
import pickle as pkl
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_dense_adj
import torch
import os
from torch_geometric.data import Batch
from torch.utils.data import DataLoader


class GraphEmo(Data):

    # Create a graph for
        # single person
        # single movie
        # single timepoint

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, adj = None, movie=None, subject=None, timestamp_tr=None):
        
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
        self.timestamp_tr = timestamp_tr
        self.adj = adj


class DatasetEmo(Dataset):

    def __init__(self, data_path):

        super(DatasetEmo, self).__init__()

        self.all_graphs = []
        self.all_labels = []

        for file in os.listdir(data_path):
            file_path = os.path.join(data_path, file)  # Construct the full path

            graph = torch.load(file_path)

            self.all_graphs.append(graph)
            self.all_labels.append(graph.y)


        self.n_samples = len(self.all_graphs)

    def __getitem__(self, index):
        graph = self.all_graphs[index]
        return graph, self.all_labels[index]
    
    def __len__(self):
        return self.n_samples


class DataLoaderEmo():

    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __iter__(self):
        # Iterate through the dataset in batches

        indices = list(range(len(self.dataset))) #[0, 1, 2, ...]
        
        if self.shuffle:
            # Shuffle the indices using torch.randperm
            indices = torch.randperm(len(self.dataset)).tolist()

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices] # Attentin, each elemt of the lsit is a tuple (graph, label)
            
            batched_graphs, batched_labels = zip(*batch_data)  # Unzip the batch into graphs and labels
            batched_labels = torch.tensor(batched_labels, dtype=torch.long)

            # yield an output in the tuple (bathced_graph, batched labels)
            yield batched_graphs, batched_labels





