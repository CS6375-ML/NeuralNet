# Neural Network Analysis

This project implements a neural network to analyze datasets from the UCI Machine Learning Repository. The script preprocesses the data, then trains and evaluates a neural network with various hyperparameters.

## Prerequisites

- Python 3.6 or later
- The libraries listed in `requirements.txt`

## Installation

1. Clone the repository or download the source code.
2. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the neural network analysis, execute the `neuralNet.py` script from your terminal:

```bash
python neuralNet.py
```

The script will perform the following steps:
1. Fetch a dataset by its ID from the UCI Machine Learning Repository.
2. Preprocess the data, which includes handling missing values and converting categorical features to numerical ones.
3. Split the data into training and testing sets.
4. Standardize the features.
5. Train and evaluate multiple neural network models using different combinations of activation functions, learning rates, and numbers of hidden layers.
6. Output the training and test accuracy and loss for each model.
7. Generate a plot showing the accuracy history for all models.

