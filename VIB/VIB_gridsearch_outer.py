import subprocess
import json
import os
from itertools import product
from pathlib import Path
from datetime import datetime

# CHECK GPUS PERIODICALLY: watch -n 1 nvidia-smi


# Define hyperparameter grid
param_grid = {

    "type_prediction": ["all_emo"], #all_emo, only ione emo e.e. "1"
    "type_dataset": ["balanced"], #balanced, unbalanced
    "how_many_movies": [14], #how mna movies use to test the model, 1, 8, ...
    "gpu_id" : ["1"], #0,1,2,3
    "use_one_sub": [False], #if to use only one subject data


    # DATASET PARAMETERS
    "num_classes": [13],  # Attention depends on "type_prediction": 13 --> all_emo
    "type_labels": ["single"],  
    "batch_size": [8],
    "test_batch_size": [8],
    "percentage_train": [0.8],
    "percentage_val": [0.0],
    "test_train_splitting_mode": ["Vertical"],
    "window_half_size": [5, 10], #10, 12
    "node_feat": ["symmetricwindow"],
    "initial_adj_method": ["clique_edgeAttr_FC_window", "FN_edgeAttr_FC_window"], #FN_edgeAttr_FC_window
    "FN": ["Limbic"],
    "thr_FC": [0.7],

    # VIB PARAMETERS
    "dataset_name": ["EMOTION"],  # Fixed value
    "backbone": ["GAT", "GIN"],
    "hidden_dim": [32, 64],
    "num_layers": [3],
    "graph_type": ["KNN"],
    "top_k": [20,40],  # For KNN
    "epsilon": [0.5],  # For epsiloNN
    "graph_metric_type": ["mlp", "cosine"],
    "num_per": [13],
    "feature_denoise": [False],
    "repar": [False],
    "beta": [0.000001],
    "IB_size": [32], #16, 64
    "graph_skip_conn": [0.0],
    "graph_include_self": [False],

    # VIB TRAINING PARAMETERS
    "epochs": [10],
    "lr": [0.0001],
    "lr_decay_factor": [0.5],
    "lr_decay_step_size": [50],
    "weight_decay": [5e-5],
}

# Directory to save results
RESULTS_DIR = Path("data/results/VIB/gridsearch_json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create all combinations of hyperparameters
param_combinations = list(product(*param_grid.values()))
print(f"In total {len(param_combinations)} combinations evaluated.")
param_names = list(param_grid.keys())

# Iterate over all parameter combinations
for idx, params in enumerate(param_combinations):
    print(f"\nRunning configuration {idx + 1}/{len(param_combinations)}")
    
    # Create a dictionary of parameters for this combination
    params_dict = dict(zip(param_names, params))
    
    # Save parameters to a temporary JSON file with unique identifier
    params_file = RESULTS_DIR / f"params_{idx + 1}.json"
    with open(params_file, "w") as f:
        json.dump(params_dict, f, indent=4)
    print(f"Parameters json saved as {params_file}")

    # Call the main script
    try:
        print("Starting script inner execution...")
        result = subprocess.run(
            ["python", "VIB/VIB_train.py", "--params", str(params_file)],
            #stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            text=True
        )
        print(f"Configuration {idx + 1} completed.")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        # Log errors if subprocess fails
        if result.returncode != 0:
            print(f"Error in configuration {idx + 1}: {result.stderr}")
    except Exception as e:
        print(f"Exception occurred for configuration {idx + 1}: {e}")
