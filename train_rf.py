import os
import pandas as pd
from datetime import datetime
import traceback
import warnings
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.sparse import vstack
import joblib
from sparse_dataset import load_sparse_split

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 1000
SEED = 12345


def modelo_completo(trials, study_name, dataset_dir="./datasets"):
    """
    Random Forest con pesos + tf-idf con los splits preprocesados (train/val/test).
    Optimiza contra df_val y reserva df_test para una etapa posterior.
    """
    try:
        dataset_dir = os.path.abspath(dataset_dir)
        print(f"[{datetime.now()}] - Leyendo train/val en formato sparse desde {dataset_dir}/final_sparse")
        X_train, y_train = load_sparse_split(dataset_dir, "train")
        X_val, y_val = load_sparse_split(dataset_dir, "val")

        def cv_es_rf_objective(trial):
            rf_params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
                'max_depth': trial.suggest_categorical('max_depth', [None, 8, 12, 16, 24, 32]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'n_jobs': -1,
                'random_state': SEED,
            }

            rf_model = RandomForestClassifier(**rf_params)

            rf_model.fit(X_train, y_train)
            preds_val = rf_model.predict(X_val)
            kappa = cohen_kappa_score(y_val, preds_val)
            trial.set_user_attr("val_accuracy", accuracy_score(y_val, preds_val))
            return kappa

        # Genero estudio
        study = optuna.create_study(
            direction='maximize',
            storage=BBDD,
            study_name=f"RF_{study_name}",
            load_if_exists=True,
        )

        # Corro la optimizacion
        study.optimize(cv_es_rf_objective, n_trials=trials)

        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\\n")
        model = RandomForestClassifier(**study.best_params, n_jobs=-1, random_state=SEED)

        # Entrenamos con train+val para usar el mayor volumen posible antes de evaluar en test
        X_full = vstack([X_train, X_val], format='csr')
        y_full = np.concatenate([y_train, y_val])

        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \\n")
        model.fit(X_full, y_full)

        model_dir = os.path.join("models", "randomforest", study_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{study_name}.pkl")
        joblib.dump(model, model_path)
        print(f"[{datetime.now()}] - Se ha guardado el modelo en {model_path} \\n")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\\n{tb}")


def get_numeric_columns(df):
    """
    Devuelve una lista de columnas numéricas en el DataFrame df.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove("target")
    return numeric_cols


if __name__ == "__main__":
    modelo_completo(trials=TRIALS, study_name="modelo_completo")
