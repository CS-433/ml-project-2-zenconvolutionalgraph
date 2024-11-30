
################################
### Libraries
################################
import os
#os.chdir("/home/dalai/GNN_E")
print(os.getcwd())

import sys
scripts_path = os.getcwd()
print(scripts_path)
sys.path.append(scripts_path)

from utils_models import *
from GAT_model import *

from math import ceil
import pandas as pd
import argparse
from types import SimpleNamespace
import json
from contextlib import redirect_stdout
from pathlib import Path
import matplotlib.pyplot as plt
import random

import torch
import torch.optim as optim

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)

gpu_mem()

################################
### Hyperparamters
################################

# Argument parser for JSON parameters
parser = argparse.ArgumentParser()
parser.add_argument("--params", type=str, help="Path to JSON file containing parameters.")
args_cli = parser.parse_args()

print(args_cli)

# Load parameters from JSON file
with open(args_cli.params, "r") as f:
    params = json.load(f)

# Convert the dictionary to a SimpleNamespace
args = SimpleNamespace(**params)

print(args)

gpu_mem()


################################
### Load all movies
################################

if args.type_dataset == "balanced":
    df_all_movies = pd.read_csv(f"data/processed/all_movies_labelled_13_single_balanced.csv")
if args.type_dataset == "unbalanced":
    df_all_movies = pd.read_csv(f"data/processed/all_movies_labelled_{args.num_classes}_{args.type_labels}.csv")



if args.type_prediction == "all_emo":
    pass
else: #single emo
    # all oher emo gain specific class "77"
    # problem now of unbalance
    df_all_movies.loc[~ df_all_movies.label.isin([int(x) for x in args.type_prediction]), "label"] = 77


if args.how_many_movies == 1:
    df_all_movies = df_all_movies[df_all_movies.movie.isin([0,9])]
if args.how_many_movies == 6:
    df_all_movies = df_all_movies[df_all_movies.movie.isin([0,1,2,3,5,6,7,4,9])]
else: #use all
    pass

gpu_mem()

################################
### Split in Train, Validation, Test
################################

print(f"Splitting {args.test_train_splitting_mode}...")

if args.test_train_splitting_mode == "Vertical":
    #df_train, df_test = split_train_test_vertically(
    #    df_all_movies, 
    #   test_movies_dict = {"BigBuckBunny": 2, "FirstBite": 4, "Superhero": 9})
    # In case use balanced dataset
    df_train, df_test = split_train_test_vertically(
        df_all_movies, 
        test_movies_dict = {"FirstBite": 4, "Superhero": 9, "YouAgain": 13})
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

gpu_mem()

################################
### Create Datasets
################################

with open(os.devnull, "w") as fnull:
    with redirect_stdout(fnull):
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

gpu_mem()

# Extarct the list of graphs of each dataset
graphs_list_train = dataset_train.get_graphs_list()
graphs_list_val = dataset_val.get_graphs_list()
graphs_list_test = dataset_test.get_graphs_list()

# Create a dataloader
loader_train = pyg.loader.DataLoader(graphs_list_train, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
loader_val = pyg.loader.DataLoader(graphs_list_val, batch_size=args.batch_size, num_workers=4, persistent_workers=True)
loader_test = pyg.loader.DataLoader(graphs_list_test, batch_size=args.batch_size, num_workers=4, persistent_workers=True)

#Claulte number of batches
num_batches_train = ceil(len(graphs_list_train) / args.batch_size)
num_batches_val = ceil(len(graphs_list_val) / args.batch_size)
num_batches_test = ceil(len(graphs_list_test) / args.batch_size)

print(f"There are {len(graphs_list_train)} graphs in the train set.")
print(f"There are {len(graphs_list_val)} graphs in the train set.")
print(f"There are {len(graphs_list_test)} graphs in the test set.")
print(f"N batches in train: {num_batches_train}")
print(f"N batches in val: {num_batches_val}")
print(f"N batches in test: {num_batches_test}")

gpu_mem()

# for idx, data in enumerate(graphs_list_train):
#     if data.x.shape[1] != 11:
#         print(f"Graph {idx}:")
#         print(f"  Node Features: {data.x.shape if data.x is not None else 'None'}")
#         print(f"  Edge Index: {data.edge_index.shape if data.edge_index is not None else 'None'}")
#         print(f"  Edge Attributes: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")

################################
### Instantiate GAT
################################

n_feat_per_node = graphs_list_train[0].x.shape[1]
print(f"n_feat_per_node: {n_feat_per_node}")

MyGat = GATModel(
    input_dim = n_feat_per_node, 
    hidden_dim = args.num_classes, 
    output_dim = args.num_classes, 
    num_heads = args.num_classes
)

MyGat = MyGat.to(device)

gpu_mem()

#print(next(MyGat.parameters()).device)
#print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Calculate the total number of trainable parameters
total_trainable_params = sum(p.numel() for p in MyGat.parameters() if p.requires_grad)
print(f"\nTotal number of trainable parameters: {total_trainable_params}\n")


################################
### Train GAT
################################

torch.autograd.set_detect_anomaly(True)

best_model, results = GAT_train(
    model=MyGat, 
    train_loader=loader_train, 
    test_loader=loader_test, 
    num_epochs=args.epochs, 
    learning_rate=args.lr
)

gpu_mem()

################################
### Predict Labels Using best model
################################

pred_y_test = GAT_eval(best_model, loader_test) #predocted 
y_test = [g.y.item() for g in graphs_list_test] #true labels

################################
### Save Results
################################

# Extract results
train_losses = results["train_losses"]
test_losses = results["test_losses"]
train_accs = results["train_accuracies"]
test_accs = results["test_accuracies"]
best_test_accuracy = results["best_test_accuracy"]

# Create Folder
RESULT_DIR = Path(f"data/results/GAT/{args.type_dataset}/{args.how_many_movies}/{args.type_prediction}/{int(best_test_accuracy*10000)}")
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
        "pred_y_test": pred_y_test, 
        "y_test": y_test,
    }
)

# Save dict with results
with open(os.path.join(RESULT_DIR, 'results.json'), 'w') as f:
    json.dump(results_dict, f, indent=4)

# Save best model
torch.save(best_model, os.path.join(RESULT_DIR, 'full_model.pth'))

################################
### Plot Accuracies
################################

epochs = range(1, args.epochs + 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss over Epochs")
plt.savefig(os.path.join(RESULT_DIR, 'losses.png'))








