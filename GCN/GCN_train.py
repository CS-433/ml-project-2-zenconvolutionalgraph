import sys
import os

os.chdir("/home/zhzhou/GNN_E")
print(os.getcwd())

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts'))
sys.path.append(scripts_path)
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './data'))
sys.path.append(scripts_path)
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(scripts_path)
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './GCN'))
sys.path.append(scripts_path)
from utils_models import *
from GCN_models import *


import pandas as pd
from math import ceil
import gc
from types import SimpleNamespace
from pathlib import Path
import random
import json
import argparse
# Torch Libraries
import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

import numpy as np
import pandas as pd
import torch
from math import ceil
from utils_models import *
from GCN_models import *
from torch_geometric.loader import DataLoader


#### Add arguments about the GCN model
def parse_args():
    parser = argparse.ArgumentParser(description="Train a GCN model on a cluster")
    parser.add_argument('--data-path', type=str, help="Path to the dataset")
    parser.add_argument('--batch-size', type=int, help="Batch size for DataLoader")
    parser.add_argument('--num-workers', type=int, help="Number of workers for DataLoader")
    parser.add_argument('--num-epochs', type=int, help="Number of training epochs")
    parser.add_argument('--lr', type=float, help="Learning rate for optimizer")
    parser.add_argument('--input-dim', type=int, help="Input feature dimension")
    parser.add_argument('--hidden-dim1', type=int, help="First hidden layer dimension")
    parser.add_argument('--hidden-dim2', type=int, help="Second hidden layer dimension")
    parser.add_argument('--output-dim', type=int, help="Output feature dimension (number of classes)")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help="Device for computation")
    parser.add_argument('--node_feat', type=str, help="calibration type")
    parser.add_argument('--config', type=str, help="Path to JSON configuration file")
    parser.add_argument('--initial_adj_method', type=str, help="initial_adj_method")
    return parser.parse_args()

def load_json_config(json_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON configuration file '{json_path}' not found.")
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_args()

    # Load arguments from JSON if provided
    if args.config:
        print(f"Loading configuration from {args.config}...")
        json_config = load_json_config(args.config)
        for key, value in json_config.items():
            setattr(args, key, value)

    # Validate mandatory arguments
    if not args.data_path:
        raise ValueError("The --data-path argument is required.")

    # Ensure the device is set correctly
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(args.data_path, index_col=0)
    df_train, df_test = split_train_test_vertically(df)  # Implement this function if not available

    dataset_train = DatasetEmo(df_train, device=device,FN = 'Limbic', initial_adj_method = args.initial_adj_method,node_feat=args.node_feat)
    dataset_test = DatasetEmo(df_test, device=device,FN = 'Limbic', initial_adj_method = args.initial_adj_method, node_feat=args.node_feat)

    graphs_list_train = dataset_train.get_graphs_list()
    graphs_list_test = dataset_test.get_graphs_list()

    # Set up DataLoaders
    train_loader = DataLoader(
        graphs_list_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True
    )
    test_loader = DataLoader(
        graphs_list_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    print(f"There are {len(graphs_list_train)} graphs in the train set.")
    print(f"There are {len(graphs_list_test)} graphs in the test set.")
    print(f"N batches in train: {ceil(len(graphs_list_train) / args.batch_size)}")
    print(f"N batches in test: {ceil(len(graphs_list_test) / args.batch_size)}")

    # Model setup
    print("Initializing the model...")
    model = SimpleGCN(
        input_dim=args.input_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        output_dim=args.output_dim
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    print("Starting training...")
    torch.cuda.empty_cache()
    model, results_dict = GCN_train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=args.num_epochs
    )
    print("Training completed!")

    file_name = f'data/results/GCN/GCNModel_{args.config.split('.')[0]}.pth'
    torch.save(model.state_dict(), file_name)
    with open(f'data/results/GCN/GCNModel_results_{args.config.split('.')[0]}.pkl','wb') as f:
        pkl.dump(results_dict,f)

if __name__ == "__main__":
    main()