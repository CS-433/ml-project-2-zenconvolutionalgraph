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

                    if node_feat == "symmetricwindow":
                        sizewind = 4
                        time_around = [i for i in range(timepoint - sizewind, timepoint + sizewind + 1)]
                        x = df_single_movie_sub.loc[df_single_movie_sub.timestamp_tr.isin(time_around) ,["vindex", "score", "timestamp_tr"]]
                        x = x.pivot(index= "vindex", columns = "timestamp_tr", values = "score")
                        #print(x.shape) #must be (#nodes, #feat_nodes)
                        #print(x)
                        x_matrix = torch.tensor(x.values, dtype=torch.float)


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


def split_train_test_vertically(df_all_movies, test_movies_dict = {"Sintel": 7, "TearsOfSteel": 10, "Superhero": 9}):
    
    # Extract code test movies
    movie_names = df_all_movies.movie.unique()
    test_movies = list(test_movies_dict.values())
    train_movies = [movie for movie in movie_names if movie not in test_movies]

    # Split the df with all movies in train and test
    df_train = df_all_movies[df_all_movies.movie.isin(train_movies)]
    df_test = df_all_movies[df_all_movies.movie.isin(test_movies)]

    return df_train, df_test

def split_train_test_horizontally(df_all_movies, percentage_test = 0.2, path_pickle_delay = "data/raw/labels/run_onsets.pkl", tr_len = 1.3):
    
    #Idea: I can say that one datapoint is NO in test/train set by tranforming its label in -1
    # Indeed only timepoints that have labe != -1 will be used to create a graph and thus to be predicted

    # I will split in order ot have always the final x% of the movie in the test set

    # Attnetion: still info leackage on the border

    #Attention: for same movie but differt subject the movie starts at differt time (but just few sec)
    # in this case as we ar elobisn only few labels we ignore this fact in the splitting procedure

    # Split the df with all movies in train and test
    df_train = df_all_movies.copy()
    df_test = df_all_movies.copy()
   
    # Load onset of different movies, for differt subejcts
    with open("data/raw/labels/run_onsets.pkl", "rb") as file:
        delta_time = pkl.load(file)

    movies = df_all_movies["movie"].unique()

    for movie in movies:
        # Access the dictionary of subjects for the current movie
        subject_onsets = delta_time[movie]
        # Select the first available subject (assuming we don't need to specify which one)
        first_subject = next(iter(subject_onsets))
        # Retrieve start and duration for this subject
        start_movie_sec, length_movie_sec = subject_onsets[first_subject]

        start_movie_tr = int(start_movie_sec / tr_len)
        lenght_movie_tr = int(length_movie_sec / tr_len)
        start_test_set = start_movie_tr + int(lenght_movie_tr * percentage_test)

        # put -1 in the timestamp of the test set inside  train set
        df_train.loc[(df_train.movie == movie) & (df_train.timestam_tr > start_test_set), "label"] = -1

        # do the opposite
        df_test.loc[(df_test.movie == movie) & (df_test.timestam_tr < start_test_set), "label"] = -1


    return df_train, df_test


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





