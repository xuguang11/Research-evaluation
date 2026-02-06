------

# Research on the evaluation of medical staff’s scientific research ability based on transformer-based graph convolutional networks

## Description

This repository contains a comprehensive pipeline for evaluating the scientific research ability of medical professionals. The core of the project is a hybrid neural network architecture that integrates **Fully Connected Encoders**, **Transformer Attention mechanisms**, and **Graph Convolutional Networks (GCN)**.

The framework is designed to process complex survey data, transform categorical and textual professional information into high-dimensional embeddings, and leverage both local feature correlations (via Transformers) and global sample relationships (via GCNs) to classify research proficiency into three levels: Low, Medium, and High.

------

## Dataset Information

The dataset is derived from professional surveys of medical staff (`new_science_data.xlsx`).

- **Features**: Includes demographic data (age, gender, ethnicity), professional status (unit name, technical title, department, work years), and scientific activity metrics (types of research activities, resource requirements, policy support).
- **Preprocessing**:
  - **Textual Encoding**: Uses **Word2Vec** (via `gensim`) to convert professional titles and department names into vector representations (e.g., 8-dimensional vectors for departments).
  - **Categorical Encoding**: One-hot encoding for gender, region, and training frequency.
  - **Cleaning**: Handles missing values via zero-padding, rounds float values, and corrects anomalous scoring data (clamping scores to a 0–5 range).
- **Labels**: The target variable is "Rank" (等级), categorized as Low (0), Medium (1), and High (2).

------

## Code Information

The codebase is structured into modular components:

- **`model_main.py`**: The entry point for the project. It executes 5-fold cross-validation and manages the training/testing flow.
- **`fusion_GCN.py`**: Implements the primary architecture (`Model_design`) combining Transformer encoders and GCN layers.
- **`DataWashing.py`**: A comprehensive script for data cleaning, Word2Vec training, and feature engineering.
- **`compare_fusion.py`**: Contains baseline models for performance comparison, including CNN, LSTM, MLP, ResNet, and DenseNet.
- **`init_train_test.py`**: Defines the training loops, optimizer configurations, and evaluation metrics (Accuracy, F1, AUC, etc.).
- **`options.py`**: Manages hyperparameters and command-line arguments.
- **`utils.py`**: Provides utility functions for t-SNE visualization and metric calculation.

------

## Methodology

### 1. Data Processing Pipeline

1. **Cleaning**: Filling nulls, correcting data entry errors, and normalizing scales.
2. **Feature Extraction**: Numerical features are normalized, while textual/categorical features are processed through Word2Vec or One-hot encoding to create a unified feature vector .
3. **Graph Construction**: An adjacency matrix ($X_{adj}$) is generated to represent relationships between different staff members, allowing the GCN to capture cohort-level patterns.

### 2. Model Architecture

The model utilizes a multi-stage fusion approach:

- **FC Encoder**: Initial feature projection and normalization.
- **Transformer Layer**: Captures internal dependencies within the individual’s feature set using self-attention.
- **GCN Layer**: Processes the feature representations within the context of the constructed social/professional graph.
- **Bi-LSTM**: Refines the integrated features for final classification into the three ability tiers.

------

## Requirements

To run this project, you need **Python 3.8+** and the following libraries:

- **Deep Learning**: `torch`, `torchvision`
- **Data Science**: `numpy`, `pandas`, `scipy`, `scikit-learn`
- **NLP/Embedding**: `gensim` (for Word2Vec)
- **Visualization**: `matplotlib`, `cv2`
- **Utilities**: `tqdm`, `thop` (for FLOPs calculation), `openpyxl`

------

## Usage Instructions

### 1. Data Preparation

Place your source data (`new_science_data.xlsx`) in the root directory. Run the cleaning script to generate the processed files:

Bash

```
python DataWashing.py
```

### 2. Training and Evaluation

The system uses 5-fold cross-validation. To start the training process with the default Transformer-GCN architecture:

Bash

```
python model_main.py --model_name Transformer_GCN_Exp --batch_size 10 --lr 0.0005
```

### 3. Hyperparameter Tuning

You can modify parameters in `options.py` or via command line:

- `--hgcn_dim`: Adjust the hidden dimensions of the GCN layers.
- `--dropout_rate`: Change dropout to prevent overfitting.
- `--input_size`: Adjust based on the output of your data washing (default is 214).

### 4. Viewing Results

Results, including model weights (`.pt`) and prediction pickles (`.pkl`), will be saved in the `./model_save_file` directory. The console will output the average Accuracy, Precision, Recall, and AUC across all five folds.