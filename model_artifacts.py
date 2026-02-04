import os
import joblib
import xgboost as xgb
import json
from typing import Any


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Generic joblib save/load
def save_joblib(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_joblib(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    # fallback: search for the basename under any artifacts folder
    basename = os.path.basename(path)
    found = _find_artifact_file(basename)
    if found:
        return joblib.load(found)
    return None


# Sklearn pipeline helpers
def save_sklearn_pipeline(pipeline, dirpath, name):
    ensure_dir(dirpath)  # Ensure the directory exists
    path = os.path.join(dirpath, f"{name}.joblib")
    save_joblib(pipeline, path)
    return path


def load_sklearn_pipeline(dirpath, name):
    path = os.path.join(dirpath, f"{name}.joblib")
    pl = load_joblib(path)
    if pl is not None:
        return pl
    # fallback: search by filename
    found = _find_artifact_file(f"{name}.joblib")
    if found:
        return load_joblib(found)
    return None


# XGBoost helpers
def save_xgb_model(model: xgb.Booster, dirpath, name):
    ensure_dir(dirpath)
    path = os.path.join(dirpath, f"{name}.xgb")
    model.save_model(path)
    return path


def load_xgb_model(dirpath, name):
    path = os.path.join(dirpath, f"{name}.xgb")
    if not os.path.exists(path):
        # fallback: search for file under artifacts folders
        found = _find_artifact_file(f"{name}.xgb")
        if found:
            path = found
        else:
            return None
    model = xgb.Booster()
    model.load_model(path)
    return model


# Torch helpers: save state_dict and arbitrary metadata
def save_torch_state(state_dict, dirpath, name):
    ensure_dir(dirpath)
    path = os.path.join(dirpath, f"{name}.pth")
    import torch
    torch.save(state_dict, path)
    return path


def load_torch_state(dirpath, name):
    path = os.path.join(dirpath, f"{name}.pth")
    if os.path.exists(path):
        import torch
        return torch.load(path)
    # fallback search
    found = _find_artifact_file(f"{name}.pth")
    if found:
        import torch
        return torch.load(found)
    return None


# json helper for small metadata
def save_json(obj, dirpath, name):
    ensure_dir(dirpath)
    path = os.path.join(dirpath, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(obj, f)
    return path


def load_json(dirpath, name):
    path = os.path.join(dirpath, f"{name}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    found = _find_artifact_file(f"{name}.json")
    if found:
        with open(found, 'r') as f:
            return json.load(f)
    return None


# Helper: search for artifact file under any 'artifacts' directory upwards from cwd and module dir
def _find_artifact_file(filename: str):
    # absolute path already
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    # search upward from current working dir and module dir
    starts = [os.getcwd(), os.path.abspath(os.path.dirname(__file__))]
    seen = set()
    for start in starts:
        cur = start
        while True:
            if cur in seen:
                break
            seen.add(cur)
            artifacts_dir = os.path.join(cur, 'artifacts')
            # direct match under artifacts
            candidate = os.path.join(artifacts_dir, filename)
            if os.path.exists(candidate):
                return candidate
            # search recursively inside artifacts for filename
            if os.path.isdir(artifacts_dir):
                for root, dirs, files in os.walk(artifacts_dir):
                    if filename in files:
                        return os.path.join(root, filename)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return None
