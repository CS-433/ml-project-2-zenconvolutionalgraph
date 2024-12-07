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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Parallel, delayed


def custom_corrcoef(x_matrix, thr = None):
    # Convert x_matrix to numpy if it's a PyTorch tensor
    if isinstance(x_matrix, torch.Tensor):
        x_matrix = x_matrix.numpy()

    # Identify valid (non-NaN and non-zero) values
    valid_mask = ~np.isnan(x_matrix) & (x_matrix != 0)

    # Identify rows that are fully zero or entirely invalid
    row_is_zero = np.all(x_matrix == 0, axis=1)

    # Compute the number of valid elements per row
    valid_counts = valid_mask.sum(axis=1)

    # Compute sums of valid elements per row
    row_sums = np.nansum(np.where(valid_mask, x_matrix, 0), axis=1)

    # Safely compute row means
    row_means = np.zeros_like(row_sums)
    valid_rows = valid_counts > 0  # Only rows with valid data
    row_means[valid_rows] = row_sums[valid_rows] / valid_counts[valid_rows]

    # Subtract the row means (mean-centering)
    centered = np.where(valid_mask, x_matrix - row_means[:, None], 0)

    # Compute dot products and norms
    dot_products = np.dot(centered, centered.T)
    norms = np.sqrt(np.sum(centered**2, axis=1))
    norm_products = np.outer(norms, norms)

    # Handle cases where norm_products is zero
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings temporarily
        corr_matrix = np.where(norm_products > 0, dot_products / norm_products, 0)

    # Set correlations involving rows that are fully zero to 0
    corr_matrix[row_is_zero, :] = 0
    corr_matrix[:, row_is_zero] = 0

    # Retain only positive values
    #corr_matrix[corr_matrix <= 0] = 0
    #corr_matrix[corr_matrix <= 0] = abs(corr_matrix[corr_matrix <= 0])

    if thr != None:
        corr_matrix[abs(corr_matrix) < thr] = 0

    # Dno use self loops
    corr_matrix = corr_matrix - np.diag(np.diag(corr_matrix))

    return corr_matrix




class DatasetEmo_fast():

    def __init__(self,
                df, #df with mvoies to use
                node_feat = "singlefmri", #"singlefmri", "symmetricwindow", "pastwindow"
                initial_adj_method = "clique",
                    # "clique"
                    #FC dynamic:  "fcmovie", "fcwindow"
                    #FN (subcorticla with clique): "FN_const" "FN_edgeAttr_FC_window" "FN_edgeAttr_FC_movie"
                FN = None, #['Vis' 'SomMot' 'DorsAttn' 'SalVentAttn' 'Limbic' 'Cont' 'Default' 'Sub']
                FN_paths = "data/raw/FN_raw",
                device = "cpu", # I want to move data in GPU ONLY during batch
                sizewind = 4,
                verbose = False, # Vervose will print the fucntional connecity matrix, ONLY of the first graph
                thr_FC = None, #thr to use for functional connectiovity
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
        #for clique FN
        df_FN = pd.read_csv(os.path.join(FN_paths, f"FN_{FN}.csv")) #remember that the "Sub" FN is always present
        nodes_in_FN = df_FN[df_FN.is_in_FN == 1].vindex.values # Extract nodes that are in the FN subset
        self.nodes_in_FN = nodes_in_FN
        self.nodes_not_in_FN = df_FN[~(df_FN.is_in_FN == 1)].vindex.values
        edge_index_clique_FN = torch.combinations(torch.tensor(nodes_in_FN), r=2).t()  # Pairwise combinations
        self.edge_index_clique_FN = torch.cat([edge_index_clique_FN, edge_index_clique_FN.flip(0)], dim=1)  # Add both directions
        self.edge_attr_clique_FN = torch.ones(self.edge_index_clique_FN.size(1), 1)  # 1 attribute per edge
        # For only self loops (indentity adjancy matrix)
        self.edge_index_I = torch.tensor([[i for i in range(414)],  # Source nodes
                           [i for i in range(414)]],  # Target nodes
                          dtype=torch.long)
        self.edge_attr_I = torch.ones((414, 1))
        

        # Ectarct movies
        movies = df["movie"].unique()
        print(f"Movies in this df: {movies}")

        for movie in movies:

            #df of the data to builf a single grapg
            df_single_movie = df[df.movie == movie]

            subjects = df_single_movie["id"].unique()

            print(movie)

            for sub in tqdm(subjects):

                print(sub)

                df_single_movie_sub = df_single_movie[df_single_movie.id == sub]

                #timepoint to rpedict
                timepoints = df_single_movie_sub[df_single_movie_sub.label != -1]["timestamp_tr"].unique()
                #print(len(timepoints), timepoints)

                #ATTENTION: ORDER ROWS BY VINDEX, SO SURE THAT INDEX ARE INCREASINGLY
                df_single_movie_sub = df_single_movie_sub.sort_values(by="vindex")

                list_small = Parallel(n_jobs=50)(delayed(self.parallelization_timepoint_per_movie_sub)(df_single_movie_sub, movie, sub, tp, sizewind, node_feat, initial_adj_method, FN, thr_FC, verbose) for tp in timepoints)

                print(list_small)
                print(list_small)

                graph_list_small = [x[0] for x in list_small]
                graph_list_info_small = [x[1] for x in list_small]

                self.graphs_list += graph_list_small
                self.graphs_list_info += graph_list_info_small

    def get_graphs_list(self):
        return self.graphs_list
    
    def get_graphs_list_info(self):
        return self.graphs_list_info



    def parallelization_timepoint_per_movie_sub(
        self,
        df_single_movie_sub,
        movie, 
        sub, 
        timepoint,
        sizewind = 5,
        node_feat = "symmetricwindow", 
        initial_adj_method = "clique_edgeAttr_FC_window", 
        FN = None, 
        thr_FC = 0.7, 
        verbose = False
    ):
        
        # Select data of single timepoint
        df_single_movie_sub_timepoint = df_single_movie_sub[df_single_movie_sub.timestamp_tr == timepoint]

        #NODE FEAT
        if node_feat == "singlefmri":
            x = df_single_movie_sub_timepoint[["vindex", "score"]]
            x_matrix = np.array(x["score"]).reshape(-1, 1)
            #print(x_matrix.shape) #must be (#nodes, #feat_nodes)
            x_matrix = torch.tensor(x_matrix, dtype=torch.float)

        if node_feat == "symmetricwindow":
            time_around = [i for i in range(timepoint - sizewind, timepoint + sizewind + 1)]
            x = df_single_movie_sub.loc[df_single_movie_sub.timestamp_tr.isin(time_around) ,["vindex", "score", "timestamp_tr"]]
            x = x.pivot(index= "vindex", columns = "timestamp_tr", values = "score")
            #print(x.shape) #must be (#nodes, #feat_nodes)
            #print(x)
            x_matrix = torch.tensor(x.values, dtype=torch.float)


        #NODE CONNECTIVITY
        functional_connectivity_matrix = None # it is necessary for verbose
            #attnetion df alredy ordered before by vindex
        if initial_adj_method == "clique":
            # Each node is connected to every other node (both directions)
            edge_index = self.edge_index_clique_414
            # Create edge_attr with value 1 for each edge
            edge_attr = self.edge_attr_clique_414  # 1 attribute per edge
        elif initial_adj_method == "clique_edgeAttr_FC_window":
            edge_index = self.edge_index_clique_414
            # compute and put in the correct order functional conecotuty
            functional_connectivity_matrix = custom_corrcoef(x_matrix, thr=thr_FC)    
            edge_attr = []
            for i in range(edge_index.size(1)):  # Loop over each edge
                node1, node2 = edge_index[:, i].numpy()  # Extract node1 and node2 for the current edge
                edge_value = functional_connectivity_matrix[node1, node2]  # Extract the correlation value
                edge_attr.append(edge_value)
            #make tensor
            edge_index = edge_index# already a tensor
            edge_attr = torch.tensor(edge_attr)    
        elif initial_adj_method == "FN_const_1":
            # use only nodes of a specific FN, and use as attr of the edges the scalar 1
            assert FN != None, "Want to create connectivity with FN, but not specific FN has been defined"
            edge_index = self.edge_index_clique_FN
            edge_attr = self.edge_attr_clique_FN  
            # put the features of all OTHERS nodes to 0
            # x_matrix --> (#nodes, #feat_nodes) --> put the correpsoding roes to 0
            x_matrix[self.nodes_not_in_FN] = 0
        elif initial_adj_method == "FN_edgeAttr_FC_window":
            # use only nodes of a specific FN, and use as attr of the edges the FC calculated inside the window
            assert FN != None, "Want to create connectivity with FN, but not specific FN has been defined"
            edge_index = self.edge_index_clique_FN
            # put the features of all OTHERS nodes to 0
            x_matrix[self.nodes_not_in_FN] = 0
            #print(self.nodes_not_in_FN)
            #print(x_matrix)
            #print(x_matrix.shape)
            # Edge attr build with FC of the current window --> no connecoty between region non in FN (removed rows from x_matrix)
            #functional_connectivity_matrix = np.corrcoef(x_matrix) #(correlation between nodes' time series)
            functional_connectivity_matrix = custom_corrcoef(x_matrix, thr=thr_FC)                      
            
            #print(functional_connectivity_matrix)
            #print(functional_connectivity_matrix.shape)
            # Iterate over each edge in edge_index and extract the corresponding value from the matrix
            edge_attr = []
            for i in range(edge_index.size(1)):  # Loop over each edge
                node1, node2 = edge_index[:, i].numpy()  # Extract node1 and node2 for the current edge
                edge_value = functional_connectivity_matrix[node1, node2]  # Extract the correlation value
                edge_attr.append(edge_value)
            #make tensor
            edge_index = edge_index# already a tensor
            edge_attr = torch.tensor(edge_attr)
        elif initial_adj_method == "I": # only self loops
            edge_index = self.edge_index_I
            edge_attr = self.edge_attr_I

        #GRAPH LABEL
        y = df_single_movie_sub_timepoint["label"].unique()[0]
        y = torch.tensor(y, dtype=torch.long)

        if verbose and (functional_connectivity_matrix is not None):
            # In case we have a FN, use only those nodes
            # if FN is not None:
            #     zero_rows = np.all(functional_connectivity_matrix == 0, axis=1)
            #     zero_cols = np.all(functional_connectivity_matrix == 0, axis=0)
            #     functional_connectivity_matrix = functional_connectivity_matrix[~zero_rows][:, ~zero_cols]
            # Print functional connectivity calculated
            print(functional_connectivity_matrix.shape)
            plt.imshow(functional_connectivity_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label="Connectivity Strength")
            plt.xlabel("Region Index"); plt.ylabel("Region Index")
            plt.title(f"Functional Connectivity Matrix, Sub {sub}, Movie {movie}, time {timepoint}")
            # Check if grpah is connected with speicifc thr
            G_temp = nx.from_numpy_array(functional_connectivity_matrix)
            print(nx.is_connected(G_temp))
            # Print hist of values in functional connectivity
            plt.figure()
            flattened_values = functional_connectivity_matrix.flatten()
            print(len(flattened_values))
            plt.hist(flattened_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Histogram of Matrix Values')
            plt.xlabel('Value')
            #plt.xlim(-0.1, 0.1)
            plt.yscale("log")
            plt.ylabel('Frequency')
            plt.show()   
            #return  

        graph = Data(x=x_matrix, edge_index=edge_index, edge_attr=edge_attr, y = y)
        info_graph = [movie, sub, timepoint, y]

        return graph, info_graph



###############################


class DatasetEmo():

    def __init__(self,
                df, #df with mvoies to use
                node_feat = "singlefmri", #"singlefmri", "symmetricwindow", "pastwindow"
                initial_adj_method = "clique",
                    # "clique"
                    #FC dynamic:  "fcmovie", "fcwindow"
                    #FN (subcorticla with clique): "FN_const" "FN_edgeAttr_FC_window" "FN_edgeAttr_FC_movie"
                FN = None, #['Vis' 'SomMot' 'DorsAttn' 'SalVentAttn' 'Limbic' 'Cont' 'Default' 'Sub']
                FN_paths = "data/raw/FN_raw",
                device = "cpu", # I want to move data in GPU ONLY during batch
                sizewind = 4,
                verbose = False, # Vervose will print the fucntional connecity matrix, ONLY of the first graph
                thr_FC = None, #thr to use for functional connectiovity
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
        #for clique FN
        df_FN = pd.read_csv(os.path.join(FN_paths, f"FN_{FN}.csv")) #remember that the "Sub" FN is always present
        nodes_in_FN = df_FN[df_FN.is_in_FN == 1].vindex.values # Extract nodes that are in the FN subset
        self.nodes_in_FN = nodes_in_FN
        self.nodes_not_in_FN = df_FN[~(df_FN.is_in_FN == 1)].vindex.values
        edge_index_clique_FN = torch.combinations(torch.tensor(nodes_in_FN), r=2).t()  # Pairwise combinations
        self.edge_index_clique_FN = torch.cat([edge_index_clique_FN, edge_index_clique_FN.flip(0)], dim=1)  # Add both directions
        self.edge_attr_clique_FN = torch.ones(self.edge_index_clique_FN.size(1), 1)  # 1 attribute per edge
        # For only self loops (indentity adjancy matrix)
        self.edge_index_I = torch.tensor([[i for i in range(414)],  # Source nodes
                           [i for i in range(414)]],  # Target nodes
                          dtype=torch.long)
        self.edge_attr_I = torch.ones((414, 1))
        

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

                for timepoint in tqdm(timepoints, desc=f"Processing {movie} {sub}", unit="timepoint"):

                    #print(f"Creating the graph {movie} {sub} {timepoint-timepoints[0]}/{len(timepoints)}")

                    # Select data of single timepoint
                    df_single_movie_sub_timepoint = df_single_movie_sub[df_single_movie_sub.timestamp_tr == timepoint]
                                         
                    #NODE FEAT
                    if node_feat == "singlefmri":
                        x = df_single_movie_sub_timepoint[["vindex", "score"]]
                        x_matrix = np.array(x["score"]).reshape(-1, 1)
                        #print(x_matrix.shape) #must be (#nodes, #feat_nodes)
                        x_matrix = torch.tensor(x_matrix, dtype=torch.float)

                    if node_feat == "symmetricwindow":
                        time_around = [i for i in range(timepoint - sizewind, timepoint + sizewind + 1)]
                        x = df_single_movie_sub.loc[df_single_movie_sub.timestamp_tr.isin(time_around) ,["vindex", "score", "timestamp_tr"]]
                        x = x.pivot(index= "vindex", columns = "timestamp_tr", values = "score")
                        #print(x.shape) #must be (#nodes, #feat_nodes)
                        #print(x)
                        x_matrix = torch.tensor(x.values, dtype=torch.float)


                    #NODE CONNECTIVITY
                    functional_connectivity_matrix = None # it is necessary for verbose
                        #attnetion df alredy ordered before by vindex
                    if initial_adj_method == "clique":
                        # Each node is connected to every other node (both directions)
                        edge_index = self.edge_index_clique_414
                        # Create edge_attr with value 1 for each edge
                        edge_attr = self.edge_attr_clique_414  # 1 attribute per edge
                    elif initial_adj_method == "clique_edgeAttr_FC_window":
                        edge_index = self.edge_index_clique_414
                        # compute and put in the correct order functional conecotuty
                        functional_connectivity_matrix = custom_corrcoef(x_matrix, thr=thr_FC)    
                        edge_attr = []
                        for i in range(edge_index.size(1)):  # Loop over each edge
                            node1, node2 = edge_index[:, i].numpy()  # Extract node1 and node2 for the current edge
                            edge_value = functional_connectivity_matrix[node1, node2]  # Extract the correlation value
                            edge_attr.append(edge_value)
                        #make tensor
                        edge_index = edge_index# already a tensor
                        edge_attr = torch.tensor(edge_attr)    
                    elif initial_adj_method == "FN_const_1":
                        # use only nodes of a specific FN, and use as attr of the edges the scalar 1
                        assert FN != None, "Want to create connectivity with FN, but not specific FN has been defined"
                        edge_index = self.edge_index_clique_FN
                        edge_attr = self.edge_attr_clique_FN  
                        # put the features of all OTHERS nodes to 0
                        # x_matrix --> (#nodes, #feat_nodes) --> put the correpsoding roes to 0
                        x_matrix[self.nodes_not_in_FN] = 0
                    elif initial_adj_method == "FN_edgeAttr_FC_window":
                        # use only nodes of a specific FN, and use as attr of the edges the FC calculated inside the window
                        assert FN != None, "Want to create connectivity with FN, but not specific FN has been defined"
                        edge_index = self.edge_index_clique_FN
                        # put the features of all OTHERS nodes to 0
                        x_matrix[self.nodes_not_in_FN] = 0
                        #print(self.nodes_not_in_FN)
                        #print(x_matrix)
                        #print(x_matrix.shape)
                        # Edge attr build with FC of the current window --> no connecoty between region non in FN (removed rows from x_matrix)
                        #functional_connectivity_matrix = np.corrcoef(x_matrix) #(correlation between nodes' time series)
                        functional_connectivity_matrix = custom_corrcoef(x_matrix, thr=thr_FC)                      
                        
                        #print(functional_connectivity_matrix)
                        #print(functional_connectivity_matrix.shape)
                        # Iterate over each edge in edge_index and extract the corresponding value from the matrix
                        edge_attr = []
                        for i in range(edge_index.size(1)):  # Loop over each edge
                            node1, node2 = edge_index[:, i].numpy()  # Extract node1 and node2 for the current edge
                            edge_value = functional_connectivity_matrix[node1, node2]  # Extract the correlation value
                            edge_attr.append(edge_value)
                        #make tensor
                        edge_index = edge_index# already a tensor
                        edge_attr = torch.tensor(edge_attr)
                    elif initial_adj_method == "I": # only self loops
                        edge_index = self.edge_index_I
                        edge_attr = self.edge_attr_I

                    #GRAPH LABEL
                    y = df_single_movie_sub_timepoint["label"].unique()[0]
                    y = torch.tensor(y, dtype=torch.long)

                    if verbose and (functional_connectivity_matrix is not None):
                        # In case we have a FN, use only those nodes
                        # if FN is not None:
                        #     zero_rows = np.all(functional_connectivity_matrix == 0, axis=1)
                        #     zero_cols = np.all(functional_connectivity_matrix == 0, axis=0)
                        #     functional_connectivity_matrix = functional_connectivity_matrix[~zero_rows][:, ~zero_cols]
                        # Print functional connectivity calculated
                        print(functional_connectivity_matrix.shape)
                        plt.imshow(functional_connectivity_matrix, cmap='viridis', aspect='auto')
                        plt.colorbar(label="Connectivity Strength")
                        plt.xlabel("Region Index"); plt.ylabel("Region Index")
                        plt.title(f"Functional Connectivity Matrix, Sub {sub}, Movie {movie}, time {timepoint}")
                        # Check if grpah is connected with speicifc thr
                        G_temp = nx.from_numpy_array(functional_connectivity_matrix)
                        print(nx.is_connected(G_temp))
                        # Print hist of values in functional connectivity
                        plt.figure()
                        flattened_values = functional_connectivity_matrix.flatten()
                        print(len(flattened_values))
                        plt.hist(flattened_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
                        plt.title('Histogram of Matrix Values')
                        plt.xlabel('Value')
                        #plt.xlim(-0.1, 0.1)
                        plt.yscale("log")
                        plt.ylabel('Frequency')
                        plt.show()   
                        #return  


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


def split_train_val_test_horizontally(df_all_movies, percentage_train=0.8, percentage_val=0.0, path_pickle_delay="data/raw/labels/run_onsets.pkl", path_movie_title_mapping="data/raw/labels/category_mapping_movies.csv", tr_len=1.3):
    """
    Splits the movie data into train, validation, and test sets based on sequential timing.
    The split is done based on the movie's timeline, ensuring no randomization.

    Args:
    - df_all_movies: DataFrame containing all movie data with timestamps and labels.
    - percentage_train: Proportion of the movie's data to be used for training.
    - percentage_val: Proportion of the movie's data to be used for validation.
    - path_pickle_delay: Path to the pickle file containing the onsets of the movies.
    - path_movie_title_mapping: Path to the CSV file mapping movie titles to numeric ids.
    - tr_len: Length of each time step in seconds (TR length).

    Returns:
    - df_train: DataFrame with updated labels for training data.
    - df_val: DataFrame with updated labels for validation data.
    - df_test: DataFrame with updated labels for test data.
    """
    # Load the onset times for different subjects in different movies
    with open(path_pickle_delay, "rb") as file:
        delta_time = pkl.load(file)

    # Load mapping of movie title to movie ID
    df_movie_mapping = pd.read_csv(path_movie_title_mapping)

    # Create empty DataFrames for train, validation, and test
    df_train = df_all_movies.copy()
    df_val = df_all_movies.copy()
    df_test = df_all_movies.copy()

    # Loop through each movie to perform the sequential split
    movies = df_all_movies["movie"].unique()

    for movie in movies:
        # Retrieve the movie string name
        movie_str = df_movie_mapping[df_movie_mapping.movie == movie]["movie_str"].values[0]
        
        # Access the dictionary of subjects for this movie
        subject_onsets = delta_time[movie_str]
        
        # Assume we are working with the first subject
        first_subject = next(iter(subject_onsets))
        
        # Retrieve the start time and duration of the movie for this subject
        start_movie_tr, length_movie_tr = subject_onsets[first_subject]

        # Add delay
        start_movie_tr += 4 #4TR
        
        # Define the splitting points based on the percentages
        end_train_set = start_movie_tr + int(length_movie_tr * percentage_train)
        end_val_set = end_train_set + int(np.ceil(length_movie_tr * percentage_val))

        print(f"\nMovie: {movie_str}")
        print(f"  Start Time (TR)+4: {start_movie_tr}")
        print(f"  Total Length (TR): {length_movie_tr}")
        print(f"  Train End (TR): {end_train_set}")
        print(f"  Validation End (TR): {end_val_set}")
        print(f"  Movie End (TR): {start_movie_tr + length_movie_tr}")
        
        # Train set: Data before the train split point
        df_train.loc[(df_train.movie == movie) & (df_train.timestamp_tr > end_train_set), "label"] = -1
        
        # Validation set: Data between the train and validation split points
        df_val.loc[(df_val.movie == movie) & (df_val.timestamp_tr <= end_train_set), "label"] = -1
        df_val.loc[(df_val.movie == movie) & (df_val.timestamp_tr > end_val_set), "label"] = -1

        
        # Test set: Data after the validation split point
        df_test.loc[(df_test.movie == movie) & (df_test.timestamp_tr <= end_val_set), "label"] = -1

    return df_train, df_val, df_test


def split_train_test_rest_classification(df_all_movies, df_rest):

    df_all_movies = df_all_movies.copy()
    df_rest = df_rest.copy()

    # chage the label, now they should be binary
        # 0 = rest
        # 1 = movie
    df_rest.loc[df_rest.label != -1, "label"] = 0 #-1 indicates timepoitns to not classifify
    df_all_movies.loc[df_all_movies.label != -1, "label"] = 1
    
    # Take a single movie, alredy checjed that isnde there is a similar number of timepotis to classify as in rest
    df_single_movie = df_all_movies[df_all_movies.movie == 0]

    df_merge = pd.concat([df_single_movie, df_rest])

    # Create train and test
        #Attnetion : they are the same df
        # the only differce is that the column label will assume -1 in difert ways
    # split horizontally
    thr_hor = 350
    df_train = df_merge.copy()
    df_train.loc[df_train.timestamp_tr > thr_hor, "label"] = -1
    df_test = df_merge.copy()
    df_test.loc[df_test.timestamp_tr <= thr_hor, "label"] = -1

    # how many classificable timepoitn soin each df
    print("Classificable timepoints in train and test")

    print(df_train[(df_train.label != -1) & (df_train.id == 1) & (df_train.vindex == 0)]["label"].value_counts().sum())
    print(df_train[(df_train.id == 1) & (df_train.vindex == 0)]["label"].value_counts())

    print(df_test[(df_test.label != -1) & (df_test.id == 1) & (df_test.vindex == 0)]["label"].value_counts().sum())
    print(df_test[(df_test.id == 1) & (df_test.vindex == 0)]["label"].value_counts())

    return df_train, df_test


def create_feature_label_tensors_for_FNN(df, sizewind=4):
    X = []
    y = []
    
    # Loop through unique movies in the dataset
    movies = df["movie"].unique()
    print(f"Movies in this df: {movies}")

    for movie in movies:
        df_single_movie = df[df.movie == movie]
        subjects = df_single_movie["id"].unique()

        for sub in subjects:
            df_single_movie_sub = df_single_movie[df_single_movie.id == sub]
            # Timepoints to predict
            timepoints = df_single_movie_sub[df_single_movie_sub.label != -1]["timestamp_tr"].unique()
            # Order rows by 'vindex'
            df_single_movie_sub = df_single_movie_sub.sort_values(by="vindex")

            for timepoint in timepoints:
                print(f"Processing movie: {movie}, subject: {sub}, timepoint: {timepoint - timepoints[0]}/{len(timepoints)}")

                # Select data for a symmetric window around the timepoint
                time_around = [i for i in range(timepoint - sizewind, timepoint + sizewind + 1)]
                x = df_single_movie_sub.loc[df_single_movie_sub.timestamp_tr.isin(time_around), ["vindex", "score", "timestamp_tr"]]
                x = x.pivot(index="vindex", columns="timestamp_tr", values="score")
                x_matrix = torch.tensor(x.values, dtype=torch.float)

                # Label
                label = df_single_movie_sub[df_single_movie_sub.timestamp_tr == timepoint]["label"].unique()[0]
                y_value = torch.tensor(label, dtype=torch.long)
                # Append the feature and label tensors to lists
                X.append(x_matrix)
                y.append(y_value)
    
    # Concatenate the list of feature and label tensors into final tensors
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    return X, y

def gpu_mem():
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Memory allocated on the current GPU device
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # Memory reserved (cached) by PyTorch on the current GPU device
        reserved_memory = torch.cuda.memory_reserved(device)

        # Print memory in bytes, you can divide by (1024**2) to convert to MB
        print(f"\nMemory Allocated: {allocated_memory / (1024**2):.2f} MB")
        print(f"Memory Reserved: {reserved_memory / (1024**2):.2f} MB\n")
    else:
        print("No GPU available")





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





