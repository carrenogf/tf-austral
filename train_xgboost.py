import os
import numpy as np
from datetime import datetime
import traceback
import warnings
import optuna
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.sparse import vstack
import joblib
from xgboost import XGBClassifier
from sparse_dataset import load_sparse_split


warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 1000
SEED = 12345


def modelo_completo(trials, study_name, dataset_dir="./datasets"):
    """
    XGBoost con pesos + tf-idf con los splits preprocesados (train/val/test).
    Optimiza contra df_val y reserva df_test para una etapa posterior.
    """
    try:
        if XGBClassifier is None:
            raise ImportError(
                "No se pudo importar xgboost. Instalalo con: pip install xgboost"
            )

        dataset_dir = os.path.abspath(dataset_dir)
        print(f"[{datetime.now()}] - Leyendo train/val en formato sparse desde {dataset_dir}/final_sparse")
        X_train, y_train = load_sparse_split(dataset_dir, "train")
        X_val, y_val = load_sparse_split(dataset_dir, "val")

        n_classes = int(np.unique(y_train).size)

        def cv_es_xgb_objective(trial):
            xgb_params = {
                'objective': 'multi:softprob',
                'num_class': n_classes,
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 1400, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 12.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': SEED,
                'n_jobs': -1,
            }

            xgb_model = XGBClassifier(**xgb_params)

            xgb_model.fit(X_train, y_train)
            preds_val = xgb_model.predict(X_val)
            kappa = cohen_kappa_score(y_val, preds_val)
            trial.set_user_attr("val_accuracy", accuracy_score(y_val, preds_val))
            return kappa

        # Genero estudio
        study = optuna.create_study(
            direction='maximize',
            storage=BBDD,
            study_name=f"xgboost_{study_name}",
            load_if_exists=True,
        )

        # Corro la optimizacion
        study.optimize(cv_es_xgb_objective, n_trials=trials)

        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\\n")

        best_model = XGBClassifier(
            **study.best_params,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=SEED,
            n_jobs=-1,
        )

        # Entrenamos con train+val para usar el mayor volumen posible antes de evaluar en test
        X_full = vstack([X_train, X_val], format='csr')
        y_full = np.concatenate([y_train, y_val])

        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \\n")
        best_model.fit(X_full, y_full)

        model_dir = os.path.join("models", "xgboost", study_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{study_name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"[{datetime.now()}] - Se ha guardado el modelo en {model_path} \\n")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\\n{tb}")


if __name__ == "__main__":
    modelo_completo(trials=TRIALS, study_name="modelo_completo")
