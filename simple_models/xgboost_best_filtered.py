import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
try:
    from model_artifacts import save_xgb_model, load_xgb_model, save_joblib, load_joblib, ensure_dir
except ModuleNotFoundError:
    import sys, os
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from model_artifacts import save_xgb_model, load_xgb_model, save_joblib, load_joblib, ensure_dir
import os
import json


class XGBoostPredictor:
    def __init__(self):
        # Configuration
        self.FILTERED = True
        self.INPUT_METRICS = ['MSIZE', 'BSIZE', 'NUM_NODES']
        self.TARGET_METRIC = 'EXEC_TIME'
        self.N_SPLITS = 5

        # XGBoost Params
        self.params = {
            'eval_metric': 'mae',
            'eta': 0.05,
            'max_depth': 4,
            'seed': 42,
            'subsample': 0.8,
            'lambda': 1.0
        }
        self.num_boost_round = 1000

        # Storage for ensemble
        self.models = []
        self.scalers_x = []
        self.scalers_y = []
        self.is_trained = False
        # Artifacts directory stored next to the module so loading works from other CWDs
        self.artifact_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'xgboost_best_filtered')

    def save_artifacts(self):
        """Save ensemble models and scalers for later loading."""
        ensure_dir(self.artifact_dir)
        meta = {
            'n_splits': self.N_SPLITS,
            'params': self.params,
            'num_boost_round': self.num_boost_round
        }
        # save each fold
        for i, model in enumerate(self.models):
            save_xgb_model(model, self.artifact_dir, f'model_fold_{i}')
            save_joblib(self.scalers_x[i], os.path.join(self.artifact_dir, f'scaler_x_fold_{i}.joblib'))
            save_joblib(self.scalers_y[i], os.path.join(self.artifact_dir, f'scaler_y_fold_{i}.joblib'))
        # save metadata
        with open(os.path.join(self.artifact_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

    def load_artifacts(self):
        """Load ensemble models and scalers if present."""
        if not os.path.exists(self.artifact_dir):
            return False
        meta_path = os.path.join(self.artifact_dir, 'meta.json')
        if not os.path.exists(meta_path):
            return False
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        # quick sanity check
        if meta.get('n_splits') != self.N_SPLITS:
            return False

        models = []
        scalers_x = []
        scalers_y = []
        for i in range(self.N_SPLITS):
            m = load_xgb_model(self.artifact_dir, f'model_fold_{i}')
            sx = load_joblib(os.path.join(self.artifact_dir, f'scaler_x_fold_{i}.joblib'))
            sy = load_joblib(os.path.join(self.artifact_dir, f'scaler_y_fold_{i}.joblib'))
            if m is None or sx is None or sy is None:
                return False
            models.append(m)
            scalers_x.append(sx)
            scalers_y.append(sy)

        self.models = models
        self.scalers_x = scalers_x
        self.scalers_y = scalers_y
        self.is_trained = True
        return True

    def train(self, data_file='filtered_dataset.csv'):
        """Loads data and trains the K-Fold ensemble."""
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)

        # Preprocessing (Calculate INPUT_SIZE for filtering)
        df['INPUT_SIZE'] = (4 * (df['BSIZE'] ** 2)) * (df['MSIZE'] ** 2) * 2

        # Filter indices
        if self.FILTERED:
            idx_train = df.index[df['INPUT_SIZE'] <= 4632608768].tolist()
        else:
            idx_train = df.index.tolist()

        # Prepare Matrices
        X_full = df[self.INPUT_METRICS].copy()
        X_full['NUM_NODES'] = X_full['NUM_NODES'] - 1  # Standardize nodes

        X_train_pool = X_full.values[idx_train]
        y_train_pool = df[self.TARGET_METRIC].values[idx_train]

        # Reset storage
        self.models = []
        self.scalers_x = []
        self.scalers_y = []

        # K-Fold Training
        kfold = KFold(n_splits=self.N_SPLITS, shuffle=True, random_state=42)
        print(f"Starting training on {len(idx_train)} rows...")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_pool)):
            # 1. Split (ora usiamo anche val_idx)
            X_train, X_val = X_train_pool[train_idx], X_train_pool[val_idx]
            y_train, y_val = y_train_pool[train_idx], y_train_pool[val_idx]

            # 2. Scale (fit solo su TRAIN, transform su VAL)
            sc_x = StandardScaler()
            sc_y = StandardScaler()

            X_train_sc = sc_x.fit_transform(X_train)
            X_val_sc = sc_x.transform(X_val)

            # Scaliamo le y per il training
            y_train_sc = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_sc = sc_y.transform(y_val.reshape(-1, 1)).flatten()

            # 3. DMatrix (creiamo anche quella di validazione)
            dtrain = xgb.DMatrix(X_train_sc, label=y_train_sc)
            dval = xgb.DMatrix(X_val_sc, label=y_val_sc)

            # 4. Train con Early Stopping
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'eval')],  # Monitora l'errore qui
                early_stopping_rounds=50,  # Si ferma se non migliora per 50 round
                verbose_eval=False
            )

            # 5. Save Artifacts
            self.models.append(model)
            self.scalers_x.append(sc_x)
            self.scalers_y.append(sc_y)

        self.is_trained = True
        print("XGBoost Ensemble Training complete.")
        try:
            self.save_artifacts()
            print(f"Artifacts saved to {self.artifact_dir}")
        except Exception:
            pass

    def predict(self, msize, bsize, num_nodes):
        """Predicts output by averaging results from all 5 fold models."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        # 1. Prepare Input (Apply the -1 logic)
        input_data = np.array([[msize, bsize, num_nodes - 1]])

        fold_predictions = []

        # 2. Predict using all stored models
        for i in range(self.N_SPLITS):
            # Scale input
            X_scaled = self.scalers_x[i].transform(input_data)
            dtest = xgb.DMatrix(X_scaled)

            # Predict (returns scaled value)
            y_pred_scaled = self.models[i].predict(dtest)

            # Inverse scale output
            y_pred_real = self.scalers_y[i].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            fold_predictions.append(y_pred_real[0])

        # 3. Return Average
        return np.mean(fold_predictions)

    def predict_batch(self, X_batch):
        """Predicts a batch of inputs using the ensemble."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        fold_predictions = np.zeros(len(X_batch))

        for i in range(self.N_SPLITS):
            X_scaled = self.scalers_x[i].transform(X_batch)
            dtest = xgb.DMatrix(X_scaled)

            y_pred_scaled = self.models[i].predict(dtest)
            y_pred_real = self.scalers_y[i].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            fold_predictions += y_pred_real

        return fold_predictions / self.N_SPLITS

    def update_csv(self, csv_file='models_comparison_RES.csv'):
        """Updates the target CSV with predictions from this model."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        print(f"Updating {csv_file}...")
        df = pd.read_csv(csv_file)

        # Prepare inputs for batch prediction
        X_test = df[self.INPUT_METRICS].copy()
        X_test['NUM_NODES'] = X_test['NUM_NODES'] - 1

        predictions = self.predict_batch(X_test)

        # Save columns
        col_name = "XGBOOST_FILTERED"
        err_name = "XGBOOST_FILTERED_%ERROR"

        df[col_name] = predictions
        df[err_name] = abs((df[self.TARGET_METRIC] - df[col_name]) / df[self.TARGET_METRIC]) * 100

        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

def exec_prediction(num_nodes, msize, bsize):
    predictor = XGBoostPredictor()
    try:
        loaded = predictor.load_artifacts()
    except Exception:
        loaded = False
    if not loaded:
        predictor.train('filtered_dataset.csv')
    return predictor.predict(msize=msize, bsize=bsize, num_nodes=num_nodes)

if __name__ == "__main__":
    predictor = XGBoostPredictor()

    # 1. Train
    if not predictor.load_artifacts():
        predictor.train('models_comparison_RES.csv')

    # 2. Update CSV
    predictor.update_csv('models_comparison_RES.csv')

    # 3. Test Single Prediction
    test_pred = predictor.predict(msize=12000, bsize=8, num_nodes=32)
    print(f"Test Prediction: {test_pred:.4f}")