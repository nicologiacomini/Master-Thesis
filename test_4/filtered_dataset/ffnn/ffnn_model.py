import torch.nn as nn
import torch
import joblib
import numpy as np
import json
import os

# ffn path
MODEL_PATH_FFN = os.path.join(os.path.dirname(__file__), 'model_ffnn.pth')
SCALER_X_PATH_FFN = os.path.join(os.path.dirname(__file__), 'scaler_X_ffnn.pkl')
SCALER_Y_PATH_FFN = os.path.join(os.path.dirname(__file__), 'scaler_y_ffnn.pkl')
USL_PARAMETERS = os.path.join(os.path.dirname(__file__), 'usl_parameters-filtered.json')


# Define the FFN model class
class ResidualNN(nn.Module):
    def __init__(self, architecture, activation):
        super(ResidualNN, self).__init__()
        # Ensure this is 3, matching your trained model (NUM_NODES, MSIZE, BSIZE)
        self.layer_1 = nn.Linear(3, architecture[0])
        self.layer_2 = nn.Linear(architecture[0], architecture[1])
        self.output_layer = nn.Linear(architecture[1], 1)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = self.output_layer(x)
        return x

def load_ffn_model():
    # Model definition here must match the class above
    model_ffnn = ResidualNN([64, 64], nn.ReLU())
    model_ffnn.load_state_dict(torch.load(MODEL_PATH_FFN))
    model_ffnn.eval()
    scaler_X = joblib.load(SCALER_X_PATH_FFN)
    scaler_y = joblib.load(SCALER_Y_PATH_FFN)
    return model_ffnn, scaler_X, scaler_y

def calculate_usl_time(n_nodi, msize, alpha, beta, baseline_time):
    usl_factor = n_nodi / (1 + alpha * (n_nodi - 1) + beta * n_nodi * (n_nodi - 1))
    usl_time = (baseline_time * msize) / usl_factor
    return usl_time

def run_ffnn(n_workers, msize, bsize):
    ffn_model, ffn_scaler_X, ffn_scaler_y = load_ffn_model()

    n_workers -= 1
    input_size = (4 * (bsize ** 2)) * (msize ** 2) * 2

    parameters = json.load(open(USL_PARAMETERS, 'r'))
    alpha = parameters['alpha']
    beta = parameters['beta']
    baseline_time = parameters['baseline_time']

    X_input = np.array([[n_workers, msize, bsize]])
    X_scaled_ffn = ffn_scaler_X.transform(X_input)
    X_tensor_ffn = torch.tensor(X_scaled_ffn, dtype=torch.float32)

    usl_time = calculate_usl_time(n_workers, msize, alpha, beta, baseline_time)

    # Predict with FFN
    ffn_model.eval()
    with torch.no_grad():
        y_pred_scaled_ffn = ffn_model(X_tensor_ffn).numpy()
    y_residual_pred_ffn = ffn_scaler_y.inverse_transform(y_pred_scaled_ffn).flatten()[0]
    predicted_time_ffn = usl_time + y_residual_pred_ffn

    return float(predicted_time_ffn)
