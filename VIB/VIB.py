import os
os.chdir("/home/dalai/GNN_E")
print(os.getcwd())

import sys
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts'))
sys.path.append(scripts_path)
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../VIB'))
sys.path.append(scripts_path)

from models import *
from utils_models import *
from train_eval import *
import gsl


import pandas as pd
from math import ceil
import gc
from types import SimpleNamespace
from pathlib import Path
import random

import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

np.random.seed(42)
random.seed(42)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)


# # Hyperparamters

# from VIB param_parser.py
args = SimpleNamespace(

    ### DATASET PARAMETERS
    num_classes=13, #number of emotions to predict
    type_labels="single", #how the labels are encoded (single value, multi class one hot found with thr, ...)
    batch_size=32,
    test_batch_size=32,
    percentage_train=0.8,
    percentage_val=0.0,
    test_train_splitting_mode="Vertical", #How to split between train and test
        # Vertical
        # Horizontal
    window_half_size=4, # Size of the window to use to create initial feautures
    node_feat="symmetricwindow", #how to create the node feautres
    initial_adj_method="clique", # hwo to initialize adge attr and edge connections
        # Clique
        # ...
    FN="Limbic", #functional method in case the iniial grpah is a subset of nodes
    
    ### VIB PARAMETERS
    dataset_name="EMOTION",
    backbone="GCN", # After graph leaning, VIB uses this backbone to precit the mu and std vectors
        # GAT, GIN, GCN
    hidden_dim=16, #hidden dim of the backbone
    num_layers=2, #number layer in case backbon eins GIN
    graph_type="prob", #how the new graph is learnt
        # epsiloNN --> Nodes are connected if their similarity (or attention score) is greater than epsilon.
        # prob -->  probabilistically, where the edges are determined by a Bernoulli distribution parameterized by the attention scores.
        # KNN --> each node to its k nearest neighbors based on the attention or similarity scores.
    top_k=10, # in case the graph is learnt with KNN
    epsilon=0.3, # in case the graph is elant using threhsolding
    graph_metric_type="mlp", #how to calculte similary between nodes in strucutre alening
        # attention
        # weighted_cosine --> cosine similarity but with learnable weights for each feature dimension
        # cosine --> raw cosine similarity
        # gat_attention --> graph attention mechanism inspired by Graph Attention Networks (GAT), using pairwise scores with a leaky ReLU activation
        # kernel --> Gaussian kernel with learnable precision to compute distances
        # transformer -->  Transformer-style attention, using query-key dot products scaled by feature dimensionality for similarity computation.
        # mlp --> multi-layer perceptron (MLP) to transform features and compute pairwise similarity.
        # multi-mlp
    num_per=16, # how many perpsctive use for graph_metric_type, ex if gat_attention (how mnay heads), if multi-mlp (how mnay indepdnet mlp)
    feature_denoise = False, 
        #enables or disables feature denoising during graph learning.
        # masl useless feaures
        # the mask is learnt
        # if true only a subsampel of feat are used for graph learning
    repar=True,
    beta=0.001,# Weighting factor for the KL divergence in the VIB loss.
        #High beta: Enforces a more compact latent representation, which can lead to better generalization by removing noise but may hurt task accuracy if too much useful information is discarded.
        #Low beta: Retains more information in the latent representation, which may improve accuracy but risks overfitting or encoding noise.
    IB_size=16, # emb dimension (i.e lenght of mu and std) (nb the last layer of gnn is double this value)
    graph_skip_conn=0.0, # between [0-1], The new adjacency matrix is a linear combination of the initial adjacency matrix and the learned adjacency matrix
    graph_include_self=True, # add self loops in new adj if graph_skip_conn==0

    ### VIB TRAINING
    folds=10,
    epochs=10,
    lr=0.0001,
    lr_decay_factor=0.5,
    lr_decay_step_size=50,
    weight_decay=5e-5,
)
print(args)


# # Load df all Movies

# Load all movies with labels csv
df_all_movies = pd.read_csv(f"data/processed/all_movies_labelled_{args.num_classes}_{args.type_labels}.csv")


############
#JUST FOR PROVA: select subset of movies
#df_all_movies = df_all_movies[df_all_movies.movie.isin([0,3])]
############


# # Split in Train, Validation, Test

print(f"Splitting {args.test_train_splitting_mode}...")

if args.test_train_splitting_mode == "Vertical":
    df_train, df_test = split_train_test_vertically(
        df_all_movies, 
        test_movies_dict = {"BigBuckBunny": 2, "FirstBite": 4, "Superhero": 9}
    )
    df_val = df_train[df_train.id == 99] #make sure to be empty
elif args.test_train_splitting_mode == "Horizontal":
    df_train, df_val, df_test = split_train_val_test_horizontally(
        df_all_movies, 
        percentage_train=args.percentage_train, 
        percentage_val=args.percentage_val, #0 to not have nay val set
        path_pickle_delay="data/raw/labels/run_onsets.pkl",
        path_movie_title_mapping="data/raw/labels/category_mapping_movies.csv", 
        tr_len=1.3
    )
elif args.test_train_splitting_mode == "MovieRest":
    df_rest = pd.read_csv("data/raw/rest/Rest_compiled414_processed.csv")
    df_train, df_test = split_train_test_rest_classification(df_all_movies, df_rest)
    df_val = df_train[df_train.id == 99] #make sure to be empty


# # Create dataset (i.e. graph list)

dataset_train = DatasetEmo(
    df = df_train, #df with mvoies to use
    node_feat = args.node_feat, #"singlefmri", "symmetricwindow", "pastwindow"
    initial_adj_method = args.initial_adj_method,
        # "clique"
        #FC dynamic:  "fcmovie", "fcwindow"
        #FN (subcorticla with clique): "FN_const" "FN_edgeAttr_FC_window" "FN_edgeAttr_FC_movie"
    FN = args.FN, #['Vis' 'SomMot' 'DorsAttn' 'SalVentAttn' 'Limbic' 'Cont' 'Default' 'Sub']
    FN_paths = "data/raw/FN_raw",
    sizewind = args.window_half_size
)

dataset_val = DatasetEmo(
    df = df_val,
    node_feat = args.node_feat,
    initial_adj_method = args.initial_adj_method,
    FN = args.FN,
    FN_paths = "data/raw/FN_raw",
    sizewind = args.window_half_size
)

dataset_test = DatasetEmo(
    df = df_test,
    node_feat = args.node_feat,
    initial_adj_method = args.initial_adj_method,
    FN = args.FN,
    FN_paths = "data/raw/FN_raw",
    sizewind = args.window_half_size
)

# Extarct the list of graphs of each dataset
graphs_list_train = dataset_train.get_graphs_list()
graphs_list_val = dataset_val.get_graphs_list()
graphs_list_test = dataset_test.get_graphs_list()

print()
print(f"Number Batces Train {len(graphs_list_train)/args.batch_size}")
print(f"Number Batces Val {len(graphs_list_val)/args.batch_size}")
print(f"Number Batces Test {len(graphs_list_test)/args.batch_size}")


# # Istantiate the Model

# following VIB main.py

# Number fo features for each node
num_node_features = graphs_list_train[0].x.shape[1]
print("\nnum_node_features : %d, num_classes : %d"%(num_node_features, args.num_classes))

model = gsl.VIBGSL(
            args, 
            num_node_features, 
            args.num_classes)
print(model.__repr__())


# # Train

# Useful if the code get some strange anomaly
torch.autograd.set_detect_anomaly(True)

train_losses, train_accs, test_losses, test_accs = my_train_and_evaluate(
    train_graphs_list = graphs_list_train,
    test_graphs_list = graphs_list_test,
    model = model,
    epochs = args.epochs, 
    batch_size = args.batch_size, 
    test_batch_size = args.test_batch_size,
    lr = args.lr, 
    lr_decay_factor = args.lr_decay_factor, 
    lr_decay_step_size = args.lr_decay_step_size,
    weight_decay = args.weight_decay, 
)


# # Evaluate on test
# Wite the last epoch model.

# Extarct accuracy, learnt grpahs and predicted lablled in test set
acc_test, new_graphs_list_test, pred_y_test = my_interpretation(
        graphs_list = graphs_list_test,
        model_trained = model,
        batch_size = args.batch_size,
)

# Extarct ground.truth labels
pred_y_test = [y.item() for y in pred_y_test]
y_test = [g.y.item() for g in graphs_list_test]

# Create a dictionary where the keys are labels and the values are lists of adjacency matrices
dict_new_graphs_list_test_adj = {}
for g, label in zip(new_graphs_list_test, y_test):
    label = str(label)  # Convert the label to string
    adj = to_dense_adj(edge_index=g.edge_index, edge_attr=g.edge_attr)
    adj = adj.cpu().squeeze().numpy()  # Convert to numpy array
    if label not in dict_new_graphs_list_test_adj:
        dict_new_graphs_list_test_adj[label] = []  # Create a list for this label if it doesn't exist
    dict_new_graphs_list_test_adj[label].append(adj)


# # Save Results


RESULT_DIR = Path(f"data/results/VIB/{int(acc_test*10000)}")
os.makedirs(RESULT_DIR, exist_ok=True)

# Convert SimpleNamespace to dictionary
results_dict = vars(args)
# Create dict with all results
results_dict.update(
    {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs, 
        "acc_test": acc_test,
        "pred_y_test": pred_y_test, 
        "y_test": y_test,
    }
)
with open(os.path.join(RESULT_DIR, 'results.json'), 'w') as f:
    json.dump(results_dict, f, indent=4)

# Save test adk mayrices
np.savez_compressed(os.path.join(RESULT_DIR, 'adj_test.npz'), **dict_new_graphs_list_test_adj, labels=y_test)

# Save the entire model (architecture + weights)
torch.save(model, os.path.join(RESULT_DIR, 'full_model.pth'))


# For future Loading
#model = torch.load('full_model.pth')
#model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# # Interpretation

# Load the .npz file
data_from_npz = np.load(os.path.join(RESULT_DIR, 'adj_test.npz'))

# Check the available keys
print(data_from_npz.files)

# Access the matrices grouped by label
label = '5'  # Example label (you can loop through all or access specific labels)
adj_list = data_from_npz[label]  # List of adjacency matrices for label '5'
print(adj_list.shape)

# Access the labels stored separately
labels = data_from_npz['labels']
print(labels.shape)  # Shape of the labels

# Print the shapes of the loaded data
print(adj_list[0].shape)  # Shape of the first adjacency matrix for this label


# graphs_list_test_fear = [g for g in graphs_list_test if g.y == 5]
# print(len(graphs_list_test_fear))
# print(graphs_list_test_fear[0])

# graphs_list, new_graphs_list, pred_y = my_interpretation(
#         graphs_list = graphs_list_test_fear,
#         model_trained = model,
#         batch_size = args.batch_size,
# )

# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch

# # Assuming initial_graph and new_graph are returned from to_dense_adj and are PyTorch tensors
# # Move tensors to CPU and convert to numpy arrays
# initial_graph_np = initial_graph.cpu().squeeze().numpy()  # Move to CPU, remove singleton dimensions, and convert to numpy
# new_graph_np = new_graph.cpu().squeeze().numpy()  # Same for new_graph

# # Set up the matplotlib figure
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create 2 subplots side by side

# # Plot the initial graph adjacency matrix
# sns.heatmap(initial_graph_np, cmap='Blues', ax=ax[0], square=True, cbar=True)
# ax[0].set_title('Initial Graph Adjacency Matrix')
# ax[0].set_xlabel('Nodes')
# ax[0].set_ylabel('Nodes')

# # Plot the new graph adjacency matrix
# sns.heatmap(new_graph_np, cmap='Blues', ax=ax[1], square=True, cbar=True)
# ax[1].set_title('New Graph Adjacency Matrix')
# ax[1].set_xlabel('Nodes')
# ax[1].set_ylabel('Nodes')

# # Adjust layout
# plt.tight_layout()
# plt.show()

