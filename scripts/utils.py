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


class DatasetEmo():

    def __init__(self,
                df, #df with mvoies to use
                node_feat = "singlefmri", #"singlefmri", "symmetricwindow", "pastwindow"
                intial_adj_method = "clique",
                    # "clique"
                    #FC dynamic:  "fcmovie", "fcwindow"
                    #FN (subcorticla with clique): "$FN_const" "$FN_weighted"
                device = "cpu" # I want to move data in GPU ONLY during batch
                ):
        
        self.device = device #or ('cuda' if torch.cuda.is_available() else 'cpu')

        # the dataset is at the end just a list of grpahs
        self.graphs_list = [] #list of all the graphs
        self.graphs_list_info = [] #list of the info of each graph

        #VALUES FOR USEFUL LATER
        n_nodes = 414 # n_nodes = df_single_movie_sub["vindex"].unique()
        # for clique grpah of 414 nodes
        edge_index_clique_414 = torch.combinations(torch.arange(n_nodes), r=2).t()
        self.edge_index_clique_414 = torch.cat([edge_index_clique_414, edge_index_clique_414.flip(0)], dim=1)
        self.edge_attr_clique_414  = torch.ones(self.edge_index_clique_414.size(1), 1)  # 1 attribute per edge


        # Ectarct movies
        movies = df["movie"].unique()
        print(f"Movies in this df: {movies}")

        for movie in movies:

            #df of the data to builf a single grapg
            df_single_movie = df[df.movie == movie]

            subjects = df_single_movie["id"].unique()

            for sub in subjects:

                df_single_movie_sub = df_single_movie[df_single_movie.id == sub]

                #timepoint to rpedict
                timepoints = df_single_movie_sub[df_single_movie_sub.label != -1]["timestamp_tr"].unique()
                #print(len(timepoints), timepoints)

                #ATTENTION: ORDER ROWS BY VINDEX, SO SURE THAT INDEX ARE INCREASINGLY
                df_single_movie_sub = df_single_movie_sub.sort_values(by="vindex")

                for timepoint in timepoints:

                    print(f"Creating the graph {movie} {sub} {timepoint-timepoints[0]}/{len(timepoints)}")

                    # Select data of single timepoint
                    df_single_movie_sub_timepoint = df_single_movie_sub[df_single_movie_sub.timestamp_tr == timepoint]
                                         
                    #NODE FEAT
                    if node_feat == "singlefmri":
       
                        x = df_single_movie_sub_timepoint[["vindex", "score"]]
                        x_matrix = np.array(x["score"]).reshape(-1, 1)
                        #print(x_matrix.shape) #must be (#nodes, #feat_nodes)
                        x_matrix = torch.tensor(x_matrix, dtype=torch.float)

                    #NODE CONNECTIVITY
                        #attnetion df alredy ordered before by vindex
                    if intial_adj_method == "clique":
                        # Each node is connected to every other node (both directions)
                        edge_index = self.edge_index_clique_414
                        # Create edge_attr with value 1 for each edge
                        edge_attr = self.edge_attr_clique_414  # 1 attribute per edge

                    #GRAPH LABEL
                    y = df_single_movie_sub_timepoint["label"].unique()[0]
                    y = torch.tensor(y, dtype=torch.long)

                    #MOVE TO DEVICE
                    #x_matrix = x_matrix.clone().detach().float().to(self.device)
                    #edge_index = edge_index.to(self.device)
                    #edge_attr = edge_attr.to(self.device)
                    #y = y.to(self.device)

                    graph = Data(x=x_matrix, edge_index=edge_index, edge_attr=edge_attr, y = y)
                    info_graph = [movie, sub, timepoint, y]

                    self.graphs_list.append(graph)
                    self.graphs_list_info.append(info_graph)

    def get_graphs_list(self):
        return self.graphs_list
    
    def get_graphs_list_info(self):
        return self.graphs_list_info





# class GraphEmo(Data):

#     # Create a graph for
#         # single person
#         # single movie
#         # single timepoint

#     def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, adj = None, movie=None, subject=None, timestamp_tr=None):
        
#         # In case a specific adj is passed, used it for the connectivity
#         if adj != None:
#             # Create edge_index tensor
#             edge_index = torch.tensor(adj.nonzero().t().contiguous(), dtype=torch.long)  # Shape: [2, num_edges]
#             print(edge_index)
#             print(edge_index.shape)
#             # Create edge attributes: weights can be set as 1 or extracted from the adj_matrix if applicable
#             edge_attr = torch.tensor(adj[edge_index[0, :], edge_index[1, :]], dtype=torch.float)
#             print(edge_attr)
#             print(edge_attr.shape)
        
#         # build adj from adge attr and index 
#         if adj == None:
#             adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)
   
#         super(GraphEmo, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#         # Add attr
#         self.movie = movie
#         self.subject = subject
#         self.timestamp_tr = timestamp_tr
#         self.adj = adj


# class DatasetEmo(Dataset):

#     def __init__(self, data_path):

#         super(DatasetEmo, self).__init__()

#         self.all_graphs = []
#         self.all_labels = []

#         for file in os.listdir(data_path):
#             file_path = os.path.join(data_path, file)  # Construct the full path

#             graph = torch.load(file_path)

#             self.all_graphs.append(graph)
#             self.all_labels.append(graph.y)


#         self.n_samples = len(self.all_graphs)

#     def __getitem__(self, index):
#         graph = self.all_graphs[index]
#         return graph, self.all_labels[index]
    
#     def __len__(self):
#         return self.n_samples

# class DataLoaderEmo():

#     def __init__(self, dataset, batch_size=32, shuffle=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle


#     def __iter__(self):
#         # Iterate through the dataset in batches

#         indices = list(range(len(self.dataset))) #[0, 1, 2, ...]
        
#         if self.shuffle:
#             # Shuffle the indices using torch.randperm
#             indices = torch.randperm(len(self.dataset)).tolist()

#         for start_idx in range(0, len(indices), self.batch_size):
#             batch_indices = indices[start_idx:start_idx + self.batch_size]
#             batch_data = [self.dataset[idx] for idx in batch_indices] # Attentin, each elemt of the lsit is a tuple (graph, label)
            
#             batched_graphs, batched_labels = zip(*batch_data)  # Unzip the batch into graphs and labels
#             batched_labels = torch.tensor(batched_labels, dtype=torch.long)

#             # yield an output in the tuple (bathced_graph, batched labels)
#             yield batched_graphs, batched_labels





