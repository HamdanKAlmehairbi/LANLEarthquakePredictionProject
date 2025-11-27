# Comparative Analysis of Deep Learning Architectures for Earthquake Prediction

**AI7102: Introduction to Deep Learning - Final Project (Fall 2025)**  
*Mohamed Alrayssi, Majid Ibrahim, Hamdan Almehairbi, Mohammed Ibrahim Alblooshi*

---

## 1. Project Overview

This project provides a systematic and reproducible analysis of various deep learning architectures for the **Kaggle LANL Earthquake Prediction** competition. The goal is to predict the `time_to_failure` for laboratory-simulated seismic events based on raw acoustic data.

Our work moves beyond a simple performance comparison by conducting a comprehensive experimental sweep. We evaluate five distinct model architectures against five different optimizer and learning rate scheduler configurations to identify the most effective model-optimizer pairings. The key finding is that a **Hybrid CNN-LSTM with Attention**, trained with **SGD + Momentum and a OneCycleLR scheduler**, achieves state-of-the-art performance, demonstrating a strong synergy between architecture and optimization strategy for this noisy, non-stationary time-series problem.

## 2. Key Findings

- **Hybrid Architectures Excel:** Models combining 1D CNNs for feature extraction and LSTMs for temporal modeling consistently outperform simpler, monolithic architectures.

- **Optimizer Synergy is Critical:** The choice of optimizer is as important as the model itself. A well-tuned SGD with a modern scheduler outperformed all variants of Adam, highlighting its ability to find more generalizable solutions.

- **Image-Based Models are Costly:** The `MLPER-Inspired` approach, while innovative, was computationally expensive and did not yield competitive results under our resource constraints.

- **State-of-the-Art Performance:** Our best model achieved a validation MAE of **2.277**, which is highly competitive with the winning solutions on the Kaggle leaderboard.

## 3. Project Structure

```
.
├── data/                     # (Not included in repo) Raw train.csv and test/ folder
├── analysis_notebook.ipynb   # Jupyter notebook for EDA, result visualization, and error analysis
├── config.py                 # Centralized configuration for all parameters
├── experiment.py             # Main script for running the optimizer comparison experiment
├── features.py               # Feature engineering functions
├── image_models.py           # PyTorch definition for the image-based CNN model
├── image_transformer.py      # Spectrogram generation logic
├── models.py                 # PyTorch definitions for all sequence-based models
├── prepare_dataset.py        # One-time script to process raw data and save loaders
├── predict_submission.py     # Script to generate final submission using K-Fold CV
├── trainer.py                # Core training and evaluation loops
├── requirements.txt          # Python package dependencies
└── README.md                 # This file
```

## 4. How to Run This Project

### Step 1: Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/HamdanKAlmehairbi/LANLEarthquakePredictionProject.git
    cd LANLEarthquakePredictionProject
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data Setup

1.  Download the competition data from the [Kaggle LANL Earthquake Prediction page](https://www.kaggle.com/c/lanl-earthquake-prediction/data).

2.  Create a `data/` directory in the project root.

3.  Place `train.csv` inside the `data/` directory.

4.  Unzip `test.zip` and ensure the `test/` folder (containing `seg_....csv` files) is in the project root.

The final structure should look like this:

```
lanl_earthquake_project/
├── data/
│   └── train.csv
└── test/
    ├── seg_00030f.csv
    └── ...
```

### Step 3: Run the Full Workflow

The project is designed as a three-step pipeline.

1.  **Prepare the Dataset (Run Once):**

    This script processes the raw `train.csv`, engineers features, and saves the `DataLoader` objects and the `scaler` to disk. This avoids redundant processing in later steps.

    ```bash
    python prepare_dataset.py
    ```

2.  **Run the Optimizer Experiment:**

    This is the main experimental script. It trains all five architectures with all five optimizer configurations and saves the complete results to `training_histories.json`.

    ```bash
    python experiment.py
    ```

3.  **Generate the Final Kaggle Submission:**

    This script automatically reads the results from `training_histories.json`, identifies the best-performing model and optimizer, performs 5-Fold Cross-Validation training using that configuration, and generates an ensembled `submission.csv` file.

    ```bash
    python predict_submission.py
    ```

### Step 4: Analyze the Results

After running `experiment.py`, you can explore all results and visualizations in the Jupyter Notebook:

```bash
jupyter notebook analysis_notebook.ipynb
```

## 5. Model Architectures & Experiments

- **Architectures Tested:**

  1. 1D CNN

  2. Bidirectional LSTM

  3. Hybrid CNN-LSTM

  4. Hybrid CNN-LSTM with Attention

  5. MLPER-Inspired (Image-based CNN)

- **Optimizer Configurations Tested:**

  1. **Adam (Static):** Fixed learning rate of 0.001.

  2. **Adam + OneCycleLR:** Adaptive learning rate with a warm-up and cosine decay schedule.

  3. **AdamW + OneCycleLR:** Adam with decoupled weight decay for better regularization.

  4. **SGD + Momentum + OneCycleLR:** Classic optimizer known for good generalization.

  5. **Adam + CyclicLR:** A triangular learning rate schedule.

The results clearly show that the combination of a hybrid architecture and a well-tuned, non-adaptive optimizer like SGD yielded the best performance.

