# CS-403 Machine Learning Project 2: Emotion Classification and Network Discovery: GNNs Without Predefined Graphs 

**Graph Neural Network without predefined graphs on Emo-FilM dataset**

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Authors](#authors)


## Introduction

This project focuses on emotion classification using fMRI data collected during a movie-watching paradigm. We compare **Graph Neural Networks (GNNs)** with traditional machine learning models such as Random Forest and K-Nearest Neighbors (KNN).

While advanced models, including **Graph Attention Networks (GAT)** and **Variational Information Bottleneck (VIB)**, were explored with **Graph Structure Learning (GSL)** techniques.

This repository provides the codebase for implementing and evaluating these methods, enabling further exploration of emotion classification from neuroimaging data.

The pipeline of our project is shown as follows:

![pipeline](frmi_pipeline.png)


## Project Structure

Here is an overview of the repository organization of the project:

```bash
.
├── README.md
├── requirements.txt
├── utils_models.py
├── run.py
├── .gitignore
├── EDA
│   ├── 0_explore_dataset.ipynb
│   └── 1_create_dataset.ipynb
├── KNN
│   └── KNN1.ipynb
├── RF
│   └── RandomForest.ipynb
├── FNN
│   ├── FNN_model.py
│   ├── FNN_train.ipynb
├── GCN
│   ├── args
│   │   └── config.json
│   ├── GCN_models.py
│   ├── GCN.py
│   ├── GCN_train.py
│   ├── GNN.ipynb
├── GAT
│   ├── GAT_gridsearch.py
│   ├── GAT_model.py
│   ├── GAT_train.py
├── VIB
│   ├── backbone.py
│   ├── gsl.py
│   ├── interpretation_VIB.ipynb
│   ├── layers.py
│   ├── train_eval.py
│   ├── utils.py
│   ├── VIB_gridsearch_outer.py
│   ├── VIB.ipynb
│   └── VIB_train.py
└── data
│   ├── assets
│   ├── processed
│   ├── raw
│   │   ├── FN_raw
│   │   ├── labels
│   │   └── movies

```

## Installation

Follow these steps to set up the environment and run the project:

1. **Clone the repository**:

```bash
git clone https://github.com/8gabri8/GNN_E
cd GNN_E
```

2. **Set up a virtual enviornment**:
```
python3 -m venv myenv
source myenv/bin/activate
```

3. **Install dependencies**:
```
pip install -r requirements.txt
```

4.(optional) Install additional tools if using GPUs(CUDA, cuDNN).

## Usage

1. **Access to the dataset:**

The multimodal dataset Emo-filM is firstly released in the paper [**Emo-FilM: A multimodal dataset for affective neuroscience using naturalistic stimuli**](https://www.biorxiv.org/content/10.1101/2024.02.26.582043v1). To get access to the dataset, please apply for access from the [MIP:Lab](https://miplab.epfl.ch/) from EPFL.

2. **Make sure the dataset is strcutured in the following way:**

3. **Explanatory Data Analysis:**

Run the jupyter notebook ``./EDA/0_explore_dataset.ipynb`` for Explanatory Data Analysis to get yourself famailar with the dataset and its relative structure.

4. **Data Preprocessing**:

Run the jupyter notebook ``./EDA/1_create_dataset.ipynb`` to create a ``.csv`` file with a raw fMRI value of all the subjects of all the movies, which can be used in all the analysis of the project (detailed in the jupyter notebook). More specifically, the data prepocessing steps are:

- Merge all the labelling dataset together.  
- Choose how to extract labels for each time point from the raw scores.      The approach followed in this study: extract a single emotion for each time point by selecting the emotion with the maximum absolute score value. (Other label extraction strategies can be found in the notebook.)
- Merge all the movie datasets together.
- Remove all useless information (e.g., name of the movie) and change the dataset to a lighter version. (e.g., ``float64`` -> ``int8``)
- Align each timepoint of each subject, each movie with the corresponding label, taking into account a delay of 4 TR for the BOLD signal to elicit.
- Balance the dataset using downsampling.
- (Optional) Dataframes containing information relative to each FN (functional network) can be generated. This step is necessary only if FNs are used in the following analysis.

- **Important Notes**: Due to dataset rebalancing and the fact that the fMRI session is longer than the movie duration, some time points will need to be predicted. This situation is encoded by assigning a label of -1 to these points. Be careful when proceeding with the analysis. All scripts for different machine learning methods start their analysis from the CSV file obtained in this step.

5. **Run the model**: KNN, RF, and FNN can be run easily using their respective notebooks.

GCN, GAT, and VIB require a specific structure to execute due to the high number of hyperparameters. The model has:

``*_model.py`` (inner script): Runs the actual analysis.  
``model_gridsearch.py`` (outer script): Runs a grid search (or a single run) of the inner script. This structure provides an easy wrapper for new users who want to experiment with the hyperparameters without modifying the inner logic.

## Authors

The authors of the project are: 

- Cristiano Sartori (Cyber Security, XXXXXX)   
- Gabriele Dall’Aglio (Neuro-X, XXXXXX)  
- Zhuofu Zhou  (Financial Engineering, 370337)
