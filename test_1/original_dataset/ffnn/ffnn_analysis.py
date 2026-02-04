import torch.nn as nn
import torch
import joblib
import numpy as np
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from ffnn_model import ResidualNN

# ffn path
MODEL_PATH_FFN = os.path.join(os.path.dirname(__file__), 'model_ffnn.pth')
SCALER_X_PATH_FFN = os.path.join(os.path.dirname(__file__), 'scaler_X_ffnn.pkl')
SCALER_Y_PATH_FFN = os.path.join(os.path.dirname(__file__), 'scaler_y_ffnn.pkl')
USL_PARAMETERS = os.path.join(os.path.dirname(__file__), 'usl_parameters_ffnn.json')


def load_residuals_data():
    data_path = os.path.join(os.path.dirname(__file__), 'usl_residuals_ffnn.csv')
    data = pd.read_csv(data_path)
    X = data[['NUM_NODES', 'INPUT_SIZE']].values
    y = data['RESIDUALS'].values
    return X, y


def analyze_overfitting():
    X, y = load_residuals_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load scalers
    scaler_X = joblib.load(SCALER_X_PATH_FFN)
    scaler_y = joblib.load(SCALER_Y_PATH_FFN)

    # Scale data
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Load the model
    model = ResidualNN([64, 64], nn.Sigmoid())
    model.load_state_dict(torch.load(MODEL_PATH_FFN))
    model.eval()

    # Predict on train and test
    with torch.no_grad():
        y_train_pred_scaled = model(torch.tensor(X_train_scaled, dtype=torch.float32)).numpy().flatten()
        y_test_pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Specify the filename
    filename = "output_analysis.txt"

    # Open the file in write mode ('w'). This creates the file if it doesn't exist
    # or overwrites it if it does. 'with' ensures the file is properly closed.
    with open(filename, 'w') as f:
        # Write the MAE and RMSE metrics to the file
        f.write(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}\n")
        f.write(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}\n")

        # Check for overfitting and write the analysis to the file
        if train_mae < test_mae and train_rmse < test_rmse:
            f.write("The model may be overfitting: performs better on training data than on test data.\n")
        elif train_mae > test_mae and train_rmse > test_rmse:
            f.write("The model may be underfitting: performs poorly on both training and test data.\n")
        else:
            f.write("The model seems to generalize well.\n")

    print(f"Analysis successfully written to {filename}")

    # Plot residuals
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    colors_train = ['orange' if abs((pred - actual) / actual) > 0.1 else 'green' for pred, actual in zip(y_train_pred, y_train)]
    scatter_train = plt.scatter(y_train, y_train_pred, alpha=0.7, c=colors_train)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Actual Residuals (Train)')
    plt.ylabel('Predicted Residuals (Train)')
    plt.title('Train Set: Actual vs Predicted')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Within ±10%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Outside ±10%')
    ], loc='upper left')

    plt.subplot(1, 2, 2)
    colors_test = ['orange' if abs((pred - actual) / actual) > 0.1 else 'green' for pred, actual in zip(y_test_pred, y_test)]
    scatter_test = plt.scatter(y_test, y_test_pred, alpha=0.7, c=colors_test)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Residuals (Test)')
    plt.ylabel('Predicted Residuals (Test)')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Within ±10%'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Outside ±10%')
    ], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'overfitting_analysis.png'))


if __name__ == "__main__":
    analyze_overfitting()
