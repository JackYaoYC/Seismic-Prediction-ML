# Earthquake Prediction by Machine Learning

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Under%20Review-orange)

This repository contains the official source code for the research project: **"Bridging Synthetic and Observed Seismicity: An Interpretable Machine Learning Framework for Revealing Distinct Precursor Mechanisms"**.

This project implements a comprehensive pipeline for earthquake forecasting, combining statistical seismology with advanced machine learning and deep learning techniques.

## 🌟 Key Features

1.  **Seismic Feature Engineering**: Extracts physical and statistical features (e.g., b-value, Z-value, Time elaps, etc.) from raw catalogs using sliding windows.
2.  **Binary Classification**: Benchmarks 8 ML models (RF, XGBoost, etc.) to predict the occurrence of large earthquakes.
3.  **Model Explainability**: Uses **SHAP (SHapley Additive exPlanations)** to interpret model decisions and identify precursor features.
4.  **Magnitude Prediction (Deep Learning)**: Implements **LSTM**, **Bi-LSTM**, and **Transformer** models with automated hyperparameter tuning (**Optuna**) for precise magnitude regression.

## 📂 Project Structure

```text
Seismic-Prediction-ML/
├── data/                    # Dataset storage (Raw .dat files and Processed .csv files)
├── results/                 # Output directory for plots (.png) and model weights (.pth)
├── src/                     # Source code directory
│   ├── feature_extraction.py    # Step 1: Feature Engineering
│   ├── train_models.py          # Step 2: Binary Classification Benchmark
│   ├── explainability.py        # Step 3: SHAP Explainability Analysis
│   ├── predict_lstm.py          # Step 4a: LSTM Magnitude Prediction
│   ├── predict_bilstm.py        # Step 4b: Bi-LSTM Magnitude Prediction
│   └── predict_transformer.py   # Step 4c: Transformer Magnitude Prediction
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

## 📂 Data Availability
Due to file size limitations on GitHub, this repository contains a **Sample Dataset** to demonstrate the pipeline's functionality.

### 1. Synthetic Data (Included)
The **Synthetic Earthquake Catalog** (`data/Simulated.dat`) is included in this repository. This dataset serves as a benchmark for verifying the code pipeline and model functionality.

### 2. Real-World Data (Sources)
The observed earthquake catalogs used in this study are publicly available from their respective official agencies. Users can download the data from the following websites:

* **Sichuan-Yunnan Region**:
    * Source: China Earthquake Networks Center (CENC)
    * Website: [http://data.earthquake.cn](http://data.earthquake.cn)
* **California Region**:
    * Source: United States Geological Survey (USGS) Comprehensive Catalog
    * Website: [https://earthquake.usgs.gov/earthquakes/search/](https://earthquake.usgs.gov/earthquakes/search/)
* **Japan Region**:
    * Source: Japan Meteorological Agency (JMA) Unified Earthquake Catalog
    * Website: [https://data.jma.go.jp/eqev/data/bulletin/hypo.html](https://www.data.jma.go.jp/svd/eqev/data/bulletin/hypo.html)

*Note: After downloading, please place the raw data files into the `data/` directory and update the file paths in the scripts accordingly.*

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.12+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage Guide

### Step 1: Feature Extraction

Convert raw earthquake catalog data (e.g., .dat files) into a feature matrix for machine learning.
*Note: You can configure window size (dt, Twindow) inside the script.*

```Bash
python src/feature_extraction.py
```
#### Output: **`data/California_features.csv`**

### Step 2: Binary Classification (Prediction of Large Events)

Train and evaluate 8 different machine learning classifiers. Generates performance metrics and ROC curves.

```Bash
python src/train_models.py --data ./data/California_features.csv --output ./results
```

### Step 3: Model Explainability (SHAP Analysis)

Analyze which seismic features contribute most to the predictions using SHAP values. Requires a trained Random Forest model.

```Bash
# Default threshold M >= 6.0
python src/explainability.py --data ./data/California_features.csv --threshold 6.0
```

### Step 4: Magnitude Prediction (Deep Learning)
Train deep learning models to regress the exact magnitude of future events. These scripts include Optuna for automated hyperparameter tuning.

#### Option A: LSTM Model
```Bash
python src/predict_lstm.py --data ./data/California_features.csv --trials 50 --epochs 200
```

#### Option B: Bi-LSTM Model
```Bash
python src/predict_bilstm.py --data ./data/California_features.csv --trials 50 --epochs 200
```

#### Option C: Transformer Model (Attention Mechanism)
```Bash
python src/predict_transformer.py --data ./data/California_features.csv --trials 50 --epochs 200
```

*Note: The **`trials`** argument controls how many hyperparameter combinations Optuna will try. Higher values yield better results but take longer.*

## 📊 Outputs & Visualization

All results are automatically saved in the **`results/`** folder:

### 1.Classification & Analysis
* **`Fig1_Model_Comparison.png`**: Comparison of Accuracy, Precision, Recall, and F1-Score across 8 models.
* **`Fig2_ROC_Curve.png`**: ROC curves displaying AUC performance for Random Forest vs. XGBoost.
* **`Fig3_SHAP_Summary.png`**: Beeswarm plot showing how each feature impacts the prediction output.
* **`Fig4_SHAP_Bar.png`**: Global feature importance ranking based on mean SHAP values.

### 2.Deep Learning Magnitude Prediction
For each model (`LSTM, Bi-LSTM, Transformer`), the following files are generated:
* #### Training Process:
  **`Fig_{Model}_Loss.png`**: Training vs. Validation loss curves showing convergence.
* #### Model Performance:
  **`Fig_{Model}_Scatter.png`**: Scatter plot of Observed Magnitude vs. Predicted Magnitude.
* #### Feature Importance:
  **`Fig_{Model}_Importance.png`**: Permutation feature importance analysis for the deep learning model.
* #### Model Weights:
  **`best_{model}_model.pth`**: The best trained PyTorch model weights saved for future inference.

## 📝 Citation
If you use this code or dataset in your research, please cite our paper (currently under review):

```text
@article{Yao2026,
  title={Bridging Synthetic and Observed Seismicity: An Interpretable Machine Learning Framework for Revealing Distinct Precursor Mechanisms},
  author={Yao, Yuechen and Jia, Ke and Zhang, Yizhi and Jiang, Yifan and Deng, Kai and Zhou, Shiyong and Jiang, Changsheng},
  journal={Submitted to Computers & Geosciences},
  year={2026}
}
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
