# Comparative Analysis of Deep Learning Architectures for Earthquake Prediction

**AI7102: Introduction to Deep Learning - Final Project (Fall 2025)**  
*Mohamed Alrayssi, Majid Ibrahim, Hamdan Almehairbi, Mohammed Ibrahim Alblooshi*

---

### Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Findings](#2-key-findings)
3. [Performance Results](#3-performance-results)
4. [Project Structure](#4-project-structure)
5. [How to Run This Project](#5-how-to-run-this-project)
6. [Model Architectures & Experiments](#6-model-architectures--experiments)

---

## 1. Project Overview

This project provides a systematic and reproducible analysis of various deep learning architectures for the **Kaggle LANL Earthquake Prediction** competition. The goal is to predict the `time_to_failure` for laboratory-simulated seismic events based on raw acoustic data.

Our work moves beyond a simple performance comparison by conducting a comprehensive experimental sweep. We evaluate five distinct model architectures against five different optimizer and learning rate scheduler configurations to identify the most effective model-optimizer pairings.

## 2. Key Findings

- **Top Performing Model:** A **Hybrid CNN-LSTM with Attention**, trained with **SGD + Momentum and a OneCycleLR scheduler**, achieves the best performance, demonstrating a strong synergy between a sophisticated architecture and a well-configured optimization strategy.

- **Hybrid Architectures Excel:** Models combining 1D CNNs for feature extraction and LSTMs for temporal modeling consistently outperform simpler, monolithic architectures like the standalone 1D CNN or LSTM.

- **Optimizer Synergy is Critical:** The choice of optimizer and scheduler is as important as the model itself. A well-tuned SGD with a modern scheduler outperformed all variants of Adam for the top models, highlighting its ability to find more generalizable solutions on this noisy dataset.

- **Image-Based Models Are Computationally Expensive:** The `MLPER-Inspired` approach, while innovative, was computationally demanding and did not yield competitive results compared to the best sequence-based models.

## 3. Performance Results

The table below summarizes the best validation Mean Absolute Error (MAE) achieved by each model architecture across all optimizer experiments. The results clearly show the superiority of the hybrid attention model.

| Model Architecture         | Best Optimizer Configuration  | Best Validation MAE |
| -------------------------- | ----------------------------- | ------------------- |
| **Hybrid w/ Attention**    | `SGD_Momentum_OneCycleLR`     | **2.277**           |
| **Hybrid CNN-LSTM**        | `SGD_Momentum_OneCycleLR`     | 2.296               |
| **LSTM**                   | `Adam_OneCycleLR`             | 2.366               |
| **1D CNN**                 | `Adam_OneCycleLR`             | 2.524               |
| **MLPER-Inspired (Image)** | `Adam_StaticLR`               | 2.685               |

## 4. Project Structure

The repository is organized into a modular pipeline, separating data preparation, experimentation, and analysis.

```
.
├── data/                     # (Local directory) For raw train.csv and test/ folder
├── analysis_notebook.ipynb   # Jupyter notebook for EDA, result visualization, and error analysis
├── config.py                 # Central configuration for all parameters and paths
├── experiment.py             # Main script for running the optimizer comparison experiment
├── features.py               # Feature engineering functions
├── image_models.py           # PyTorch definition for the image-based CNN model
├── image_transformer.py      # Spectrogram generation logic
├── models.py                 # PyTorch definitions for all sequence-based models
├── prepare_dataset.py        # One-time script to process raw data and save loaders
├── predict_submission.py     # Script to generate the final submission using K-Fold CV
├── requirements.txt          # Python package dependencies
├── trainer.py                # Core training and evaluation loops
├── utils.py                  # Helper functions (e.g., set_seed, get_device)
└── README.md                 # This file
```

## 5. How to Run This Project

### Prerequisites

- **Python 3.8+**
- **Hardware:** A CUDA-enabled NVIDIA GPU is **highly recommended** due to the size of the dataset and the complexity of the models. Training on a CPU will be extremely slow.

### Step 1: Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/HamdanKAlmehairbi/LANLEarthquakePredictionProject.git
    cd LANLEarthquakePredictionProject
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data Setup

1.  Download the competition data from the [Kaggle LANL Earthquake Prediction page](https://www.kaggle.com/c/lanl-earthquake-prediction/data).

2.  Create a `data/` directory in the project root and place `train.csv` inside it.

3.  Unzip `test.zip` and ensure the `test/` folder is in the project root.

### Step 3: Workflow Options

You can choose one of the following workflows depending on your goal.

#### Option A: Analyze Existing Results (Quickest)

This option allows you to explore the results of our experiments without re-running any training.

1.  Ensure `training_histories.json` is present in the root directory.

2.  Launch the Jupyter Notebook to view all plots and analysis:

    ```bash
    jupyter notebook analysis_notebook.ipynb
    ```

#### Option B: Generate Final Submission from Existing Results

This will use the pre-computed `training_histories.json` to identify the best model and then re-train it using a robust 5-Fold Cross-Validation strategy to produce `submission.csv`. This step still requires significant time and a GPU.

1.  Run the submission script:

    ```bash
    python predict_submission.py
    ```

#### Option C: Full Reproduction (Very Long)

This workflow reproduces the entire project from scratch, including data preparation and the full optimizer experiment.

⚠️ **Warning:** Running `experiment.py` on the full dataset will take many hours, even on a powerful GPU. For a quick test, first modify `config.py` by setting `MAX_TRAIN_ROWS` to a smaller number (e.g., `60_000_000`).

1.  **Prepare the Dataset (Run Once):**

    This script processes `train.csv`, engineers features, and saves `DataLoader` objects to disk.

    ```bash
    python prepare_dataset.py
    ```

2.  **Run the Optimizer Experiment:**

    This trains all models with all optimizer configurations and saves the results to `training_histories.json`.

    ```bash
    python experiment.py
    ```

3.  **Generate the Final Kaggle Submission:**

    Once the experiment is complete, run the submission script:

    ```bash
    python predict_submission.py
    ```

## 6. Model Architectures & Experiments

#### Architectures Tested:

1.  **1D CNN:** A simple baseline convolutional model.

2.  **Bidirectional LSTM:** A standard recurrent neural network for time-series.

3.  **Hybrid CNN-LSTM:** Combines a 1D CNN for feature extraction with an LSTM for sequence modeling.

4.  **Hybrid CNN-LSTM with Attention:** An enhancement to the hybrid model that allows it to weigh the importance of different time steps.

5.  **MLPER-Inspired (Image-based CNN):** A 2D CNN that treats spectrograms of the seismic signal as images.

#### Optimizer Configurations Tested:

1.  **Adam (Static):** Fixed learning rate of 0.001.

2.  **Adam + OneCycleLR:** Adaptive learning rate with a warm-up and cosine decay schedule.

3.  **AdamW + OneCycleLR:** Adam with decoupled weight decay for better regularization.

4.  **SGD + Momentum + OneCycleLR:** Classic optimizer known for good generalization.

5.  **Adam + CyclicLR:** A triangular learning rate schedule.
