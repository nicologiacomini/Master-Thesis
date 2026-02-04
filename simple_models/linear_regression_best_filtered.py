import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# Try to import helper functions, with fallback to local path
try:
    from model_artifacts import save_sklearn_pipeline, load_sklearn_pipeline, ensure_dir
except ModuleNotFoundError:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from model_artifacts import save_sklearn_pipeline, load_sklearn_pipeline, ensure_dir


class LassoPredictor:
    def __init__(self):
        # Configuration
        self.FILTERED = True
        self.INPUTS_METRICS = ['MSIZE', 'BSIZE', 'NUM_NODES']
        self.TARGET_METRIC = 'EXEC_TIME'
        self.ALPHA = 100.0  # <--- Requirement: Alpha = 100

        self.pipeline = None
        self.is_trained = False

        # Artifact management
        # <--- Requirement: Save to 'linear_regression_best_filtered'
        self.artifact_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'linear_regression_best_filtered')

    def train(self, training_data_file='filtered_dataset.csv'):
        """Loads data, applies filtering, and trains the Ridge pipeline."""
        if not os.path.exists(training_data_file):
            raise FileNotFoundError(f"Error: '{training_data_file}' not found.")

        print(f"Loading training data from {training_data_file}...")
        df = pd.read_csv(training_data_file)

        # 1. Preprocessing (Node adjustment)
        if 'NUM_NODES' in df.columns:
            df['NUM_NODES'] = df['NUM_NODES'] - 1

        # Clean NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.INPUTS_METRICS + [self.TARGET_METRIC])

        X = df[self.INPUTS_METRICS]
        y = df[self.TARGET_METRIC]

        # 2. Define and Fit Pipeline
        # <--- Requirement: RobustScaler and Lasso
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', Lasso(alpha=self.ALPHA))
        ])

        print(f"Training with Lasso (Alpha={self.ALPHA}) and RobustScaler...")
        self.pipeline.fit(X, y)
        self.is_trained = True

        print("Lasso Training complete.")
        self.save_artifacts()

    def predict(self, msize, bsize, num_nodes):
        """Predicts execution time for a single set of inputs."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        # Prepare input dataframe (Nodes must be decremented)
        nodes_adjusted = num_nodes - 1

        x_input = pd.DataFrame(
            [[msize, bsize, nodes_adjusted]],
            columns=self.INPUTS_METRICS
        )

        prediction = self.pipeline.predict(x_input)
        return prediction[0]

    def update_csv(self, csv_file='models_comparison_RES.csv'):
        """Updates the target CSV with predictions from this model."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping CSV update.")
            return

        print(f"Updating {csv_file}...")
        df = pd.read_csv(csv_file)

        # Prepare inputs for batch prediction
        X_test = df[self.INPUTS_METRICS].copy()
        X_test['NUM_NODES'] = X_test['NUM_NODES'] - 1

        predictions = self.pipeline.predict(X_test)

        # Save columns
        col_suffix = 'filtered' if self.FILTERED else 'original'
        col_name = f"LINEAR_REGRESSION_FILTERED"
        err_name = f"LINEAR_REGRESSION_FILTERED_%ERROR"

        df[col_name] = predictions
        df[err_name] = abs((df[self.TARGET_METRIC] - predictions) / df[self.TARGET_METRIC]) * 100

        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

    def save_artifacts(self):
        """Save trained sklearn pipeline to artifact folder."""
        # Force directory creation to ensure 'linear_regression_best_filtered' exists
        if not os.path.exists(self.artifact_dir):
            os.makedirs(self.artifact_dir, exist_ok=True)

        if self.pipeline is None:
            raise RuntimeError('Nothing to save, pipeline is None')

        # Save with a specific name for this model type
        save_sklearn_pipeline(self.pipeline, self.artifact_dir, 'lasso_pipeline')
        print(f"Artifacts saved to {self.artifact_dir}")

    def load_artifacts(self):
        """Load sklearn pipeline from artifact folder if present."""
        # Load looking for 'lasso_pipeline'
        pipeline = load_sklearn_pipeline(self.artifact_dir, 'lasso_pipeline')
        if pipeline is None:
            return False
        self.pipeline = pipeline
        self.is_trained = True
        return True


def exec_prediction(num_nodes, msize, bsize):
    predictor = LassoPredictor()
    # try load artifacts first to avoid retraining
    try:
        loaded = predictor.load_artifacts()
    except Exception:
        loaded = False
    if not loaded:
        predictor.train('filtered_dataset.csv')
    return predictor.predict(msize=msize, bsize=bsize, num_nodes=num_nodes)


if __name__ == "__main__":
    predictor = LassoPredictor()
    try:
        loaded = predictor.load_artifacts()
    except Exception:
        loaded = False

    if not loaded:
        # Only trains if artifacts aren't found
        predictor.train('processed_metadata.csv')

    predictor.update_csv('models_comparison_RES.csv')