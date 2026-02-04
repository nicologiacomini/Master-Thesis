import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
try:
    from model_artifacts import save_joblib, load_joblib, ensure_dir
except ModuleNotFoundError:
    import sys, os
    base_dir = os.path.abspath(os.path.dirname(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from model_artifacts import save_joblib, load_joblib, ensure_dir


class RandomForestPredictor:
    def __init__(self):
        # Configuration
        self.FILTERED = True
        self.INPUTS_METRICS = ['MSIZE', 'BSIZE', 'NUM_NODES']
        self.TARGET_METRIC = 'EXEC_TIME'

        # Hyperparameters (Hardcoded from your script)
        self.N_ESTIMATORS = 500
        self.MAX_DEPTH = None
        self.MIN_SAMPLES_SPLIT = 2

        self.model = None
        self.is_trained = False
        # module-relative artifacts
        self.artifact_dir = os.path.join(os.path.dirname(__file__), 'artifacts', 'random_forest_best_filtered')

    def save_artifacts(self):
        ensure_dir(self.artifact_dir)
        if self.model is None:
            raise RuntimeError('Nothing to save')
        save_joblib(self.model, os.path.join(self.artifact_dir, 'random_forest_filtered.joblib'))

    def load_artifacts(self):
        model = load_joblib(os.path.join(self.artifact_dir, 'random_forest_filtered.joblib'))
        if model is None:
            return False
        self.model = model
        self.is_trained = True
        return True

    def train(self, training_data_file='processed_metadata.csv'):
        """Loads data, filters it, and trains the Random Forest."""
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

        # 3. Define and Fit Model
        self.model = RandomForestRegressor(
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            min_samples_split=self.MIN_SAMPLES_SPLIT,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X, y)
        self.is_trained = True
        print("Random Forest Training complete.")
        try:
            self.save_artifacts()
        except Exception:
            pass

    def predict(self, msize, bsize, num_nodes):
        """Predicts execution time for a single set of inputs."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        # Prepare input (Apply -1 to nodes)
        nodes_adjusted = num_nodes - 1

        x_input = pd.DataFrame(
            [[msize, bsize, nodes_adjusted]],
            columns=self.INPUTS_METRICS
        )

        prediction = self.model.predict(x_input)
        return prediction[0]

    def update_csv(self, csv_file='models_comparison_RES.csv'):
        """Updates the target CSV with predictions."""
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run .train() first.")

        print(f"Updating {csv_file}...")
        df = pd.read_csv(csv_file)

        # Prepare inputs for batch prediction
        X_test = df[self.INPUTS_METRICS].copy()
        X_test['NUM_NODES'] = X_test['NUM_NODES'] - 1

        predictions = self.model.predict(X_test)

        # Save columns
        col_suffix = 'filtered' if self.FILTERED else 'original'
        col_name = f"RANDOM_FOREST_FILTERED"
        err_name = f"RANDOM_FOREST_FILTERED_%ERROR"

        df[col_name] = predictions
        df[err_name] = abs((df[self.TARGET_METRIC] - predictions) / df[self.TARGET_METRIC]) * 100

        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

def exec_prediction(num_nodes, msize, bsize):
    predictor = RandomForestPredictor()
    try:
        loaded = predictor.load_artifacts()
    except Exception:
        loaded = False
    if not loaded:
        predictor.train('filtered_dataset.csv')
    return predictor.predict(msize=msize, bsize=bsize, num_nodes=num_nodes)

if __name__ == "__main__":
    predictor = RandomForestPredictor()

    # 1. Train
    predictor.train('filtered_dataset.csv')

    # 2. Update CSV
    predictor.update_csv('comparison_simple_models.csv')