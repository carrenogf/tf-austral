import json
import os

import numpy as np
from scipy.sparse import load_npz


SPARSE_DIRNAME = "final_sparse"


def _sparse_dir(dataset_dir):
    return os.path.join(os.path.abspath(dataset_dir), SPARSE_DIRNAME)


def load_sparse_split(dataset_dir, split):
    """
    Carga X (CSR sparse) e y (numpy) para un split: train/val/test.
    """
    base_dir = _sparse_dir(dataset_dir)
    x_path = os.path.join(base_dir, f"{split}_X.npz")
    y_path = os.path.join(base_dir, f"{split}_y.npy")

    X = load_npz(x_path)
    y = np.load(y_path)
    return X, y


def load_feature_columns(dataset_dir):
    """
    Carga la lista de columnas usadas para construir la matriz sparse final.
    """
    base_dir = _sparse_dir(dataset_dir)
    cols_path = os.path.join(base_dir, "feature_columns.json")
    with open(cols_path, "r", encoding="utf-8") as file:
        return json.load(file)
