# deep-learning-anomaly-detection-autoencoder
Deep learning based anomaly detection using autoencoders and reconstruction error.
# Deep Learning Based Anomaly Detection Using Autoencoders

## Overview

This project implements an unsupervised deep learning approach for anomaly
detection using an Autoencoder.

The model learns to reconstruct normal transactions. Anomalous transactions
produce higher reconstruction errors because their patterns differ from the
patterns learned during training.

## Dataset

Credit Card Fraud Detection Dataset

The dataset contains credit card transactions made by European cardholders.
It contains highly imbalanced normal and fraudulent transactions.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Statistics

- Total transactions: 284,807
- Normal transactions: 284,315
- Fraudulent transactions: 492
- Input features: 30
- Target variable: Class

## Methodology

The project follows the following pipeline:

1. Data preprocessing
2. Feature scaling
3. Train-test splitting
4. Training the Autoencoder using normal transactions
5. Learning latent representations
6. Computing reconstruction error
7. Selecting an anomaly threshold
8. Detecting anomalous transactions
9. Evaluating the model

## Autoencoder Architecture

Input Layer
↓
Dense Layer (32 neurons)
↓
Dropout
↓
Dense Layer (16 neurons)
↓
Latent Representation (8 neurons)
↓
Dense Layer (16 neurons)
↓
Dense Layer (32 neurons)
↓
Output Layer

## Reconstruction Error

The reconstruction error is calculated using Mean Squared Error:

Reconstruction Error =
mean((Original Input - Reconstructed Input)^2)

Transactions with reconstruction errors greater than the selected threshold
are classified as anomalies.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Seaborn

## Evaluation Metrics

The model is evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

## Project Structure

```text
├── anomaly_detection_autoencoder.ipynb
├── README.md
├── requirements.txt
├── models/
├── results/
└── .gitignore
