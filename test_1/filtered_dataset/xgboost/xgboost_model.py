import numpy as np
import xgboost as xgb
import joblib
import json
import os

# xgboost path
MODEL_PATH_XGB = os.path.join(os.path.dirname(__file__), 'model_xgboost-model.json')
SCALER_X_PATH_XGB = os.path.join(os.path.dirname(__file__), 'scaler_X_xgboost-model.pkl')
SCALER_Y_PATH_XGB = os.path.join(os.path.dirname(__file__), 'scaler_y_xgboost-model.pkl')
USL_PARAMETERS = os.path.join(os.path.dirname(__file__), 'usl_parameters_xgboost-model.json')


def load_xgboost_model():
    model_xgb = xgb.Booster()
    model_xgb.load_model(MODEL_PATH_XGB)
    scaler_X = joblib.load(SCALER_X_PATH_XGB)
    scaler_y = joblib.load(SCALER_Y_PATH_XGB)
    return model_xgb, scaler_X, scaler_y

# calculate USL time function
def calculate_usl_time(n_nodi, input_size, alpha, beta, baseline_time):
    usl_factor = n_nodi / (1 + alpha * (n_nodi - 1) + beta * n_nodi * (n_nodi - 1))
    usl_time = (baseline_time * input_size) / usl_factor
    return usl_time

def run_xgboost(n_nodes, msize, bsize):
    xgb_model, xgb_scaler_X, xgb_scaler_y = load_xgboost_model()

    n_workers = n_nodes - 1
    input_size = (4 * (bsize ** 2)) * (msize ** 2) * 2

    parameters = json.load(open(USL_PARAMETERS, 'r'))
    alpha = parameters['alpha']
    beta = parameters['beta']
    baseline_time = parameters['baseline_time']

    usl_time = calculate_usl_time(n_workers, input_size, alpha, beta, baseline_time)

    X_input = np.array([[n_workers, input_size]])
    X_scaled_xgb = xgb_scaler_X.transform(X_input)
    dmatrix_input = xgb.DMatrix(X_scaled_xgb)

    # Predict with XGBoost
    y_pred_scaled_xgb = xgb_model.predict(dmatrix_input)
    y_residual_pred_xgb = xgb_scaler_y.inverse_transform(y_pred_scaled_xgb.reshape(-1, 1)).flatten()[0]
    predicted_time_xgb = usl_time + y_residual_pred_xgb

    return float(predicted_time_xgb)
