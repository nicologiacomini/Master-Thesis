import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Try to import helper functions, with fallback
try:
    from model_artifacts import save_sklearn_pipeline, load_sklearn_pipeline, ensure_dir
except ModuleNotFoundError:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    try:
        from model_artifacts import save_sklearn_pipeline, load_sklearn_pipeline, ensure_dir
    except ImportError:
        # Dummy fallback if module is totally missing
        def ensure_dir(path):
            os.makedirs(path, exist_ok=True)


        print("Warning: model_artifacts module not found. Using local fallback.")


class PolyPredictor:
    def __init__(self):
        # Configuration
        self.FILTERED = True
        self.INPUTS_METRICS = ['MSIZE', 'BSIZE', 'NUM_NODES']
        self.TARGET_METRIC = 'EXEC_TIME'

        # Hyperparameters
        self.DEGREE = 3
        self.ALPHA = 10.0

        self.pipeline = None
        self.is_trained = False

        # Artifact directory setup
        self.artifact_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'polynomial_regression_best_filtered')

    def save_artifacts(self):
        """Save trained sklearn pipeline and version to artifact folder."""
        if self.pipeline is None:
            raise RuntimeError('Nothing to save')

        # --- FIXED: Explicitly create directory ---
        if not os.path.exists(self.artifact_dir):
            print(f"Creating directory: {self.artifact_dir}")
            os.makedirs(self.artifact_dir, exist_ok=True)

        print(f"Saving pipeline to {self.artifact_dir}...")
        save_sklearn_pipeline(self.pipeline, self.artifact_dir, 'poly_pipeline_filtered')

        print("Artifacts saved successfully.")

    def load_artifacts(self):
        """Load sklearn pipeline from artifact folder if present."""
        # Check if directory exists first to avoid unnecessary errors
        if not os.path.exists(self.artifact_dir):
            return False

        pipeline = load_sklearn_pipeline(self.artifact_dir, 'poly_pipeline_filtered')
        if pipeline is None:
            return False
        self.pipeline = pipeline
        self.is_trained = True
        print("Model loaded from artifacts.")
        return True

    def train(self, training_data_file='processed_metadata.csv'):
        """Loads data, processes it, and trains the Polynomial Ridge pipeline."""
        if not os.path.exists(training_data_file):
            raise FileNotFoundError(f"Error: '{training_data_file}' not found.")

        print(f"Loading training data from {training_data_file}...")
        df = pd.read_csv(training_data_file)

        # 1. Preprocessing (Node adjustment)
        if 'NUM_NODES' in df.columns:
            df['NUM_NODES'] = df['NUM_NODES'] - 1

        # 2. Filtering Logic
        df['INPUT_SIZE'] = (4 * (df['BSIZE'] ** 2)) * (df['MSIZE'] ** 2) * 2

        if self.FILTERED:
            df = df[df['INPUT_SIZE'] <= 4632608768]

        # Clean NaNs
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=self.INPUTS_METRICS + [self.TARGET_METRIC])

        X = df[self.INPUTS_METRICS]
        y = df[self.TARGET_METRIC]

        # 3. Define and Fit Pipeline
        self.pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=self.DEGREE, include_bias=False)),
            ('ridge', Ridge(alpha=self.ALPHA))
        ])

        print(f"Training Polynomial Ridge (Degree={self.DEGREE}, Alpha={self.ALPHA})...")
        self.pipeline.fit(X, y)
        self.is_trained = True

        # 4. Save Artifacts
        # --- FIXED: Removed the try/except/pass block so we see errors if they happen ---
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
        col_name = f"POLYNOMIAL_REGRESSION_FILTERED"
        err_name = f"POLYNOMIAL_REGRESSION-FILTERED_%ERROR"

        df[col_name] = predictions
        df[err_name] = abs((df[self.TARGET_METRIC] - predictions) / df[self.TARGET_METRIC]) * 100

        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")


def exec_prediction(num_nodes, msize, bsize):
    predictor = PolyPredictor()
    # Attempt load, if fails, train
    if not predictor.load_artifacts():
        predictor.train('filtered_dataset.csv')
    return predictor.predict(msize=msize, bsize=bsize, num_nodes=num_nodes)


if __name__ == "__main__":
    predictor = PolyPredictor()

    # Attempt load, if fails, train
    try:
        loaded = predictor.load_artifacts()
    except Exception as e:
        print(f"Load failed: {e}")
        loaded = False

    if not loaded:
        print("Starting training sequence...")
        predictor.train('processed_metadata.csv')

    # 2. Update CSV
    predictor.update_csv('models_comparison_RES.csv')