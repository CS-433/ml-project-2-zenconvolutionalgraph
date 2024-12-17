import os
#os.chdir("/home/dalai/GNN_E")
print(os.getcwd())

import sys
scripts_path = os.getcwd()
print(scripts_path)
sys.path.append(scripts_path)
gcn_path = os.path.join(os.getcwd(), 'GCN')
sys.path.append(gcn_path)
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from math import ceil
from utils_models import *
from scripts.models import *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
from GCN_models import *

data_path = '/home/dalai/GNN_E/data/processed/all_movies_labelled_13_single_balanced.csv'
df =  pd.read_csv(data_path, index_col=0)
df_single = df
df_single.head()

with open("/home/dalai/GNN_E/data/raw/labels/run_onsets.pkl", "rb") as file:
    delta_time = pkl.load(file)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
df_train, df_test = split_train_test_vertically(
        df_single
    )
    
dataset_train = DatasetEmo_fast(df_train, device=DEVICE,FN = 'Limbic',initial_adj_method='clique_edgeAttr_FC_window', node_feat="symmetricwindow")
dataset_test = DatasetEmo_fast(df_test, device = DEVICE,FN = 'Limbic',initial_adj_method='clique_edgeAttr_FC_window', node_feat="symmetricwindow")

graphs_list_train = dataset_train.get_graphs_list()
graphs_list_test = dataset_test.get_graphs_list()

from torch_geometric.loader import DataLoader
from math import ceil
batch_size = 16
train_loader = DataLoader(graphs_list_train, batch_size=batch_size, num_workers=4, persistent_workers=True)
test_loader = DataLoader(graphs_list_test, batch_size=batch_size, num_workers=4, persistent_workers=True)

num_batches_train = ceil(len(graphs_list_train) / batch_size)
num_batches_test = ceil(len(graphs_list_test) / batch_size)

print(f"There are {len(graphs_list_train)} graphs in the train set.")
print(f"There are {len(graphs_list_test)} graphs in the test set.")
print(f"N batches in train: {num_batches_train}")
print(f"N batches in test: {num_batches_test}")

from tqdm import tqdm
input_dim = 9 
hidden_dim1 = 128
hidden_dim2 = 64
output_dim = 13  # Number of classes
model = SimpleGCN(input_dim, hidden_dim1, hidden_dim2, output_dim).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, capturable=False)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
num_epochs = 50
# Training loop
#torch.cuda.empty_cache()
model, results_dict = GCN_train(model=model, optimizer=optimizer, loss_fn=loss_fn, 
                                    train_loader=train_loader, test_loader=test_loader, device=DEVICE, num_epochs=50)

with open('/home/zhzhou/GNN_E/data/results/GCN/GCNModel_result_all.pkl','wb') as f:
    pkl.dump(results_dict,f)
with open('data/results/GCN/GCNModel_model_all.pkl','wb') as f:
    pkl.dump(model.state_dict(), f)