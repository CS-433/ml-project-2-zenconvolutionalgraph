import subprocess
import json
import os
from itertools import product
from pathlib import Path
from datetime import datetime

# Define hyperparameter grid
param_grid = {

    "type_prediction": ["all_emo"], #all_emo, only ione emo e.e. "1"
    "type_dataset": ["balanced"], #balanced, unbalanced
    "how_many_movies": [13], #how mna movies use to test the model, 1, 6, ...
    "use_one_sub": [False],

    # DATASET PARAMETERS
    "num_classes": [13],  # Fixed value
    "type_labels": ["single"],  
    "batch_size": [32],
    "test_batch_size": [32],
    "percentage_train": [0.8],
    "percentage_val": [0.0],
    "test_train_splitting_mode": ["Vertical"],
    "window_half_size": [10],
    "node_feat": ["symmetricwindow"],
    "initial_adj_method": ["FN_edgeAttr_FC_window"],#, "FN_edgeAttr_FC_window"],#FN_edgeAttr_FC_window, I
    "FN": ['Limbic'],#'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Cont', 'Default', 'Sub'],

    # TRAINING PARAMETERS
    "epochs": [50],
    "lr": [0.001],
}

# Directory to save results
RESULTS_DIR = Path("data/results/GAT/gridsearch_json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create all combinations of hyperparameters
param_combinations = list(product(*param_grid.values()))
print(f"In total {len(param_combinations)} combinations evaluated.")
param_names = list(param_grid.keys())

# Iterate over all parameter combinations
for idx, params in enumerate(param_combinations):
    print(f"Running configuration {idx + 1}/{len(param_combinations)}")
    
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
            ["python", "GAT/GAT_train.py", "--params", str(params_file)],
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

