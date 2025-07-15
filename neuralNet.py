#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from ucimlrepo import fetch_ucirepo

class NeuralNet:
    def __init__(self, dataId):
        self.processed_data = None
        self.raw_input = fetch_ucirepo(id=dataId)

    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        X_raw = self.raw_input.data.features
        y_raw = self.raw_input.data.target

        df = pd.concat([X_raw, y_raw], axis=1)

        print(f"Data head: {df.head}")
        print(f"Data info: {df.info}")
        print(f"Data type: {df.dtypes}")
        print(f"Data description: {df.describe()}")
        print(f"Data nulls: {df.isnull().sum()}")

        # Handling Null Values
        print("\n--- Checking for null values in the dataset, if present remove ---")
        null_values = df.isnull().sum()
        if null_values.sum() > 0:
            null_columns = null_values[null_values > 0].index.tolist()

            for column in null_columns:
                print(f"\n--- Replacing null values with median for column {column} ---")
                # Calculate the median of 'numerical_with_outliers' and fill NaN values
                print(f"Column type: {df[column].dtype}")
                if pd.api.types.is_numeric_dtype(df[column]):
                    median_numerical_with_outliers = df[column].median()
                    df[column].fillna(median_numerical_with_outliers, inplace=True)

                if pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                    mode_categorical_col = df[column].mode()[0]
                    df[column].fillna(mode_categorical_col, inplace=True)

        # Categorical to Numerical Conversion
        print("\n--- Converting Categorical to Numerical in Combined Data ---")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"Categorical columns identified: {categorical_cols.tolist()}")
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            print(f"Combined DataFrame shape after one-hot encoding: {df.shape}")
        else:
            print("No categorical columns found for one-hot encoding.")

        self.processed_data = df
        print(f"\nProcessed data (features + target) shape: {self.processed_data.shape}")
        print(f"Processed data head:\n{self.processed_data.head()}")

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        global label
        ncols = len(self.processed_data.columns)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols - 1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Normalize the features using StandardScaler
        print("\n--- Standardizing Numerical Features ---")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features standardized.")

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer.
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]
        neurons_per_hidden_layer = 50

        all_models_loss_history = []
        model_labels = []
        results_summary = []
        colors = plt.cm.jet(np.linspace(0, 1, len(activations) * len(learning_rates) * len(max_iterations) * len(num_hidden_layers)))
        color_idx = 0

        # Combine parameters using itertools.product
        param_combinations = list(itertools.product(
            activations, learning_rates, max_iterations, num_hidden_layers
        ))

        for activation, lr, max_iteration, hl_num in param_combinations:
            hidden_layer_sizes = tuple([neurons_per_hidden_layer] * hl_num)
            label = f"Act:{activation}, LR:{lr}, Iter:{max_iteration}, Layers:{hl_num}"

            try:
                # Initialize the MLPClassifier neural network model
                nn_model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver='adam',
                    learning_rate_init=lr,
                    max_iter=max_iteration,
                    random_state=42,
                    verbose=False,  # Set to True to see training progress
                    early_stopping=True,  # Set to True to stop training if validation score is not improving
                    n_iter_no_change=10,  # Stop if validation score doesn't improve for 10 consecutive epochs
                    validation_fraction=0.1  # Proportion of training data for validation during early stopping
                )

                # Train the model with the training data
                nn_model.fit(X_train_scaled, y_train)

                # --- Using Classification Metrics ---
                y_train_pred = nn_model.predict(X_train_scaled)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                print(f"\nLabel: {label}")
                print(f"Training Accuracy: {train_accuracy:.4f}")
                print(f"Training F1-score (weighted): {train_f1:.4f}")

                y_test_pred = nn_model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')
                print(f"Test Accuracy: {test_accuracy:.4f}")
                print(f"Test F1-score (weighted): {test_f1:.4f}")
                print(f"Actual iterations: {nn_model.n_iter_}")

                results_summary.append({
                    "Activation": activation,
                    "Learning Rate": lr,
                    "Max Iterations": max_iteration,
                    "Hidden Layers": hl_num,
                    "Train Accuracy": f"{train_accuracy:.4f}",
                    "Test Accuracy": f"{test_accuracy:.4f}",
                    "Train F1": f"{train_f1:.4f}",
                    "Test F1": f"{test_f1:.4f}",
                    "Final Loss": f"{nn_model.loss_:.4f}" if hasattr(nn_model, 'loss_') else "N/A"
                })

                # Store loss history for plotting
                if nn_model.loss_curve_ is not None:
                    all_models_loss_history.append(nn_model.loss_curve_)
                    model_labels.append(label)

                # Scatter plot for Actual vs. Predicted (for the last model trained for visual check)
                if color_idx == len(colors) - 1:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_test_pred, alpha=0.6, color='red', label='Predicted')
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                             color='blue', linestyle='--', linewidth=2, label='Perfect Prediction')
                    plt.title(f'Actual vs. Predicted Values (Last Model: {label})')
                    plt.xlabel('Actual Target Value')
                    plt.ylabel('Predicted Target Value')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.figtext(0.5, 0.01, 'This is a footer for the scatter plot.', ha='center',
                                fontsize=10, color='gray')
                    plt.show()
                color_idx += 1
            except Exception as e:
                print("An error occurred!, error:", e)

        # Display the summary table
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv('hyperparameter_tuning_summary.csv', index=False)
        print("\n\n--- Hyperparameter Tuning Summary ---")
        print("Hyperparameter tuning summary saved to hyperparameter_tuning_summary.csv")
        print(summary_df.to_string())


        # Plot the model history (loss curve) for all models in a single plot
        plt.figure(figsize=(14, 8))
        for i, loss_history in enumerate(all_models_loss_history):
            plt.plot(loss_history, label=model_labels[i], color=colors[i])

        plt.title('Neural Network Training Loss History for All Models')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (Log-loss)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        return 0


if __name__ == "__main__":
    neural_network = NeuralNet(529)
    print("Preparing data...")
    neural_network.preprocess()
    neural_network.train_evaluate()
