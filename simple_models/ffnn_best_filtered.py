import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
try:
    from model_artifacts import save_torch_state, load_torch_state, save_joblib, load_joblib, save_json, load_json, ensure_dir
except ModuleNotFoundError:
    import sys, os
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from model_artifacts import save_torch_state, load_torch_state, save_joblib, load_joblib, save_json, load_json, ensure_dir
import os
import json


class ExecTimePredictor:
    def __init__(self):
        # Configuration
        self.HIDDEN_NEURONS = 64
        self.LEARNING_RATE = 0.01
        self.NUM_EPOCHS = 1000
        self.N_SPLITS = 5
        self.FILTERED = True

        # Storage for the ensemble (5 models + scalers)
        self.models = []
        self.scalers_x = []
        self.scalers_y = []
        self.is_trained = False
        # artifacts dir (module-relative so loading works from other CWDs)
        self.artifact_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'ffnn_best_filtered')

    def _build_model(self):
        """Create a fresh model with same architecture to load state_dict into."""
        model = nn.Sequential(
            nn.Linear(3, self.HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_NEURONS, 1)
        )
        return model

    def save_artifacts(self):
        ensure_dir(self.artifact_dir)
        meta = {'n_splits': self.N_SPLITS, 'hidden_neurons': self.HIDDEN_NEURONS,
                'learning_rate': self.LEARNING_RATE, 'num_epochs': self.NUM_EPOCHS}
        for i, model in enumerate(self.models):
            # save state_dict
            save_torch_state(model.state_dict(), self.artifact_dir, f'model_fold_{i}')
            # save scalers
            save_joblib(self.scalers_x[i], os.path.join(self.artifact_dir, f'scaler_x_fold_{i}.joblib'))
            save_joblib(self.scalers_y[i], os.path.join(self.artifact_dir, f'scaler_y_fold_{i}.joblib'))
        save_json(meta, self.artifact_dir, 'meta')

    def load_artifacts(self):
        if not os.path.exists(self.artifact_dir):
            return False
        meta = load_json(self.artifact_dir, 'meta')
        if meta is None or meta.get('n_splits') != self.N_SPLITS:
            return False

        models = []
        scalers_x = []
        scalers_y = []
        for i in range(self.N_SPLITS):
            state = load_torch_state(self.artifact_dir, f'model_fold_{i}')
            sx = load_joblib(os.path.join(self.artifact_dir, f'scaler_x_fold_{i}.joblib'))
            sy = load_joblib(os.path.join(self.artifact_dir, f'scaler_y_fold_{i}.joblib'))
            if state is None or sx is None or sy is None:
                return False
            model = self._build_model()
            model.load_state_dict(state)
            model.eval()
            models.append(model)
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

        # Preprocessing Logic
        df['INPUT_SIZE'] = (4 * (df['BSIZE'] ** 2)) * (df['MSIZE'] ** 2) * 2

        # Filter Training Data
        if self.FILTERED:
            idx_train = df.index[df['INPUT_SIZE'] <= 4632608768].tolist()
        else:
            idx_train = df.index.tolist()

        # Prepare X (Features) and y (Target)
        X_full = df[['MSIZE', 'BSIZE', 'NUM_NODES']].copy()
        X_full['NUM_NODES'] = X_full['NUM_NODES'] - 1  # Crucial preprocessing step

        X_train_pool = X_full.values[idx_train]
        y_train_pool = df['EXEC_TIME'].values[idx_train]

        # Reset storage
        self.models = []
        self.scalers_x = []
        self.scalers_y = []

        # K-Fold Training
        kfold = KFold(n_splits=self.N_SPLITS, shuffle=True, random_state=42)
        print(f"Starting training on {len(idx_train)} rows...")

        for fold, (train_idx, _) in enumerate(kfold.split(X_train_pool)):
            # 1. Split
            X_fold = X_train_pool[train_idx]
            y_fold = y_train_pool[train_idx]

            # 2. Scale
            sc_x = StandardScaler()
            sc_y = StandardScaler()

            X_scaled = sc_x.fit_transform(X_fold)
            y_scaled = sc_y.fit_transform(y_fold.reshape(-1, 1))

            # 3. Train Model
            X_t = torch.tensor(X_scaled, dtype=torch.float32)
            y_t = torch.tensor(y_scaled, dtype=torch.float32)

            model = nn.Sequential(
                nn.Linear(3, self.HIDDEN_NEURONS),
                nn.ReLU(),
                nn.Linear(self.HIDDEN_NEURONS, 1)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(self.NUM_EPOCHS):
                optimizer.zero_grad()
                loss = loss_fn(model(X_t), y_t)
                loss.backward()
                optimizer.step()

            model.eval()

            # 4. Save Artifacts
            self.models.append(model)
            self.scalers_x.append(sc_x)
            self.scalers_y.append(sc_y)

        self.is_trained = True
        print("Training complete.")
        try:
            self.save_artifacts()
            print(f"Artifacts saved to {self.artifact_dir}")
        except Exception:
            pass

    def predict(self, msize, bsize, num_nodes):
        """Predicts output for new inputs using the ensemble average."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        # 1. Prepare Input (Apply the -1 logic to num_nodes)
        input_data = np.array([[msize, bsize, num_nodes - 1]])

        fold_predictions = []

        # 2. Predict using all 5 folds
        with torch.no_grad():
            for i in range(self.N_SPLITS):
                # Scale input using the scaler from this fold
                X_scaled = self.scalers_x[i].transform(input_data)
                X_t = torch.tensor(X_scaled, dtype=torch.float32)

                # Predict
                y_pred_scaled = self.models[i](X_t).numpy()

                # Inverse scale output
                y_pred_real = self.scalers_y[i].inverse_transform(y_pred_scaled)
                fold_predictions.append(y_pred_real.item())

        # 3. Return Average
        return np.mean(fold_predictions)

    def update_csv(self, csv_file='models_comparison_RES.csv'):
        """Updates the target CSV with predictions from this model."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        print(f"Updating {csv_file}...")
        df = pd.read_csv(csv_file)

        # Prepare inputs for batch prediction
        X_test = df[['MSIZE', 'BSIZE', 'NUM_NODES']].copy()
        X_test['NUM_NODES'] = X_test['NUM_NODES'] - 1

        predictions = []
        for _, row in X_test.iterrows():
            pred = self.predict(msize=row['MSIZE'], bsize=row['BSIZE'], num_nodes=row['NUM_NODES'] + 1)
            predictions.append(pred)

        # Save columns
        col_name = "FFNN_BEST_FILTERED"
        err_name = "FFNN_BEST_FILTERED_%ERROR"

        df[col_name] = predictions
        df[err_name] = abs((df['EXEC_TIME'] - df[col_name]) / df['EXEC_TIME']) * 100

        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")


def exec_prediction(num_nodes, msize, bsize):
    predictor = ExecTimePredictor()
    # try load artifacts first
    try:
        loaded = predictor.load_artifacts()
    except Exception:
        loaded = False
    if not loaded:
        predictor.train()
    prediction = predictor.predict(msize=msize, bsize=bsize, num_nodes=num_nodes)
    return prediction

# Optional: Keep this to run the script standalone like before
if __name__ == "__main__":
    predictor = ExecTimePredictor()
    if not predictor.load_artifacts():
        predictor.train()

    # Example Test
    prediction = predictor.predict(msize=10000, bsize=4, num_nodes=16)
    print(f"Predicted Exec Time: {prediction}")

    # Update CSV
    predictor.update_csv('models_comparison_RES.csv')