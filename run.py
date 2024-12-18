import os
import json
import numpy as np
from math import ceil
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print(os.getcwd())

import sys
sys.path.append('GAT')

from utils_models import *
from GAT.GAT_model import GATModel, GAT_eval

def main():

    # Set random seed
    SEED = 42
    np.random.seed(SEED)

    #################################
    # Configuration Device
    #################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    #################################
    # Configuration Parameters
    #################################

    # Parse the arguments
    args = parse_arguments()

    par_dict = {
        "dataset_path": args.dataset_path,
        "FN_dir": args.FN_dir,
        "prediction_path": args.prediction_path,
        "model_path": args.model_path,

        # DATASET PARAMETERS
        "num_classes": 13,  
        "type_labels": "single",  
        "batch_size": 32,
        "test_batch_size": 32,
        "percentage_train": 0.8,
        "percentage_val": 0.0,
        "test_train_splitting_mode": "Vertical",
        "window_half_size": 4,
        "node_feat": "symmetricwindow",
        "initial_adj_method": "clique_edgeAttr_FC_window",
        "FN": "Limbic",
        "kernelize_feat": False,
        "handcrafted_feat": False,
    }

    #################################
    # Load data
    #################################

    print(f"Loading Dataset...")
    df_all_movies = pd.read_csv("data/processed/all_movies_labelled_13_single_balanced.csv")

    ################################
    ### Split in Train, Validation, Test
    ################################

    print(f"Splitting In Train Test sets...")

    # Splititng "vertically", i.e. by movies
    dict_test_movies = {"FirstBite": 4, "Superhero": 9, "YouAgain": 13}

    # dict_test_movies = {"FirstBite": 4}
    # df_all_movies = df_all_movies[df_all_movies.id == 1]

    _ , df_test = split_train_test_vertically(
        df_all_movies, 
        test_movies_dict = dict_test_movies
    )

    ################################
    ### Create Datasets of Graphs
    ################################

    # Graphs created in-time in order to save space form the hsot lab server

    # Create only test set graph for reproducibility
    print("Creating graphs for prediction (could take some minutes)...")
    dataset_test = DatasetEmo_fast(
        df = df_test,
        node_feat = par_dict["node_feat"],
        initial_adj_method = par_dict["initial_adj_method"],
        FN = par_dict["FN"],
        FN_paths = par_dict["FN_dir"],
        sizewind = par_dict["window_half_size"],
        verbose = False,
        thr_FC = 0.7, 
        kernelize_feat = par_dict["kernelize_feat"],
        handcrafted_feat = par_dict["handcrafted_feat"],
    )
   
    # Extarct the list of graphs of each dataset
    graphs_list_test = dataset_test.get_graphs_list()

    # Create a dataloader
    loader_test = pyg.loader.DataLoader(graphs_list_test, batch_size=par_dict["batch_size"], num_workers=4, persistent_workers=True)

    # Claulte number of batches
    num_batches_test = ceil(len(graphs_list_test) / par_dict["batch_size"])
    print(f"There are {len(graphs_list_test)} graphs in the test set.")
    print(f"N batches in test: {num_batches_test}")

    ################################
    ### Instantiate GAT
    ################################

    # Load the saved model weights
    print("Load Pretrained Model...")
    MyGat = torch.load(args.model_path)

    # Set the model to evaluation mode
    MyGat.eval()

    ################################
    ### Predict Labels Using best model
    ################################

    pred_y_test = GAT_eval(MyGat, loader_test) #predocted 
    y_test = [g.y.item() for g in graphs_list_test] #true labels
    accuracy = accuracy_score(y_test, pred_y_test)
    f1 = f1_score(y_test, pred_y_test, average="weighted")


    ################################
    ### Save Results
    ################################

    par_dict.update(
        {
            "test_accuracy": accuracy,
            "test_f1": f1,
            "pred_y_test": pred_y_test,
            "y_test": y_test,
        }
    )

    # Save dict with results
    with open(par_dict["prediction_path"], 'w') as f:
        json.dump(par_dict, f, indent=4)


if __name__ == "__main__":
    main()