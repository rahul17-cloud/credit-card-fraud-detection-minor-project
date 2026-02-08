
# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using advanced data preprocessing, feature engineering, and class imbalance handling techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Requirements](#requirements)
- [Author](#author)

## Project Overview

This project implements a machine learning pipeline to identify fraudulent credit card transactions from a dataset of legitimate and fraudulent transactions. The main challenge is handling the highly imbalanced dataset where fraudulent transactions represent a small fraction of all transactions.

## Dataset

**Source**: Credit Card Fraud Detection Dataset (creditcard.csv)

**Features**:
- Time-based features: Time (seconds since transaction start)
- Transaction Amount: The monetary value of the transaction
- V1-V28: Principal components derived from PCA (anonymized features)
- Class: Target variable (0 = legitimate, 1 = fraudulent)

**Characteristics**:
- Highly imbalanced dataset
- Contains 30 features
- Includes both temporal and transactional information

## Features

### Data Processing
- **Exploratory Data Analysis (EDA)**
  - Visualization of fraud vs. legitimate transactions
  - Distribution analysis of transaction amounts
  - Class imbalance assessment

### Feature Engineering
- Hourly extraction from transaction time
- Log transformation of transaction amounts
- Z-score normalization for amount values
- Time difference calculations
- Binning of transaction amounts into quantiles

### Class Imbalance Handling
- **Undersampling**: RandomUnderSampler with 0.5 sampling strategy
- **Oversampling**: SMOTE with 0.5 sampling strategy
- Comparison of both approaches

### Model Architecture
- **Preprocessing Pipeline**: 
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
- **Classification Model**: Random Forest Classifier (100 estimators)
- **Evaluation Metrics**:
  - Classification Report (Precision, Recall, F1-Score)
  - ROC AUC Score
  - Precision-Recall Curve

## Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Required Libraries
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `Credit_Card_Fraud_Detection.ipynb`

## Project Structure

```
credit-card-fraud-detection/
├── Credit_Card_Fraud_Detection.ipynb  # Main Jupyter notebook with full pipeline
├── creditcard.csv                      # Dataset (unlabeled in repo, should be added locally)
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

## Methodology

### 1. Data Loading & EDA
   - Load credit card transaction data
   - Perform initial data exploration
   - Analyze class distribution and feature statistics

### 2. Feature Engineering
   - Extract temporal features (hour of day)
   - Create derived features (log-transformed amounts, z-scores)
   - Generate quantile-based binning for amounts
   - Drop irrelevant features (Time column)

### 3. Data Preparation
   - Separate features (X) and target (y)
   - Handle missing values
   - Identify numeric and categorical features

### 4. Preprocessing
   - Apply StandardScaler to numeric features
   - Apply OneHotEncoder to categorical features
   - Create unified preprocessing pipeline

### 5. Model Training
   - Build two parallel pipelines:
     - Undersampling approach
     - Oversampling approach (SMOTE)
   - Train Random Forest models with 80-20 train-test split
   - Use stratified splitting to maintain class balance

### 6. Model Evaluation
   - Generate classification reports for both models
   - Calculate ROC AUC scores
   - Plot precision-recall curves
   - Compare model performances

## Model Evaluation

### Metrics Used
- **Precision**: Proportion of predicted frauds that are actually frauds
- **Recall**: Proportion of actual frauds correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Precision-Recall Curve**: Trade-off between precision and recall at different thresholds

### Key Performance Indicators
Both undersampling and oversampling approaches are evaluated, with the oversampling (SMOTE) model typically showing better performance for fraud detection in imbalanced datasets.

## Results

The project compares two different approaches for handling class imbalance:

| Approach | Strengths | Trade-offs |
|----------|-----------|-----------|
| **Undersampling** | Faster training, smaller dataset | Loss of information, potential underfitting |
| **Oversampling (SMOTE)** | Preserves all data, better generalization | More training time, potential overfitting |

The precision-recall curve helps visualize the optimal threshold for fraud detection based on business requirements.

## Usage

### Running the Full Pipeline
1. Open the Jupyter notebook
2. Ensure `creditcard.csv` is in the same directory
3. Run cells sequentially (Shift + Enter)
4. Review outputs, visualizations, and model metrics

### Customizing the Project
- **Change sampling strategy**: Modify `sampling_strategy` value (0-1)
- **Adjust model parameters**: Update `n_estimators` or other Random Forest hyperparameters
- **Add features**: Modify the feature engineering section
- **Change evaluation metrics**: Add additional metrics in the evaluation cell

## Requirements

```
pandas>=1.0.0
numpy>=1.18.0
matplotlib>=3.1.0
scikit-learn>=0.22.0
imbalanced-learn>=0.7.0
```

## Author

Rahul

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset source: [Credit Card Fraud Detection Dataset]
- Built with scikit-learn and imbalanced-learn libraries
- Inspired by real-world fraud detection challenges in financial systems

---

**Note**: The `creditcard.csv` file is not included in this repository due to size constraints. Please download it from the source and place it in the project directory before running the notebook.
