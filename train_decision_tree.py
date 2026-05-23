import os
import pandas as pd
from datetime import datetime
import traceback
import warnings
import optuna
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.sparse import vstack
import joblib
from sparse_dataset import load_sparse_split

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 100
SEED = 12345


def modelo_completo(trials, study_name, dataset_dir="./datasets"):
    """
    Árbol de Decisiones Simple con los splits preprocesados (train/val/test).
    Optimiza contra df_val y reserva df_test para una etapa posterior.
    """
    try:
        dataset_dir = os.path.abspath(dataset_dir)
        print(f"[{datetime.now()}] - Leyendo train/val en formato sparse desde {dataset_dir}/final_sparse")
        X_train, y_train = load_sparse_split(dataset_dir, "train")
        X_val, y_val = load_sparse_split(dataset_dir, "val")

        def cv_es_dt_objective(trial):
            dt_params = {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'random_state': SEED,
            }

            dt_model = DecisionTreeClassifier(**dt_params)

            dt_model.fit(X_train, y_train)
            preds_val = dt_model.predict(X_val)
            kappa = cohen_kappa_score(y_val, preds_val)
            trial.set_user_attr("val_accuracy", accuracy_score(y_val, preds_val))
            return kappa

        # Crear o cargar el estudio
        try:
            study = optuna.load_study(study_name=f"dt_modelo_completo_{study_name}", storage=BBDD)
            print(f"[{datetime.now()}] - Estudio '{study.study_name}' cargado.")
        except:
            study = optuna.create_study(study_name=f"dt_modelo_completo_{study_name}", storage=BBDD,
                                        direction='maximize')

        # Ejecutar optimización
        study.optimize(cv_es_dt_objective, n_trials=trials, n_jobs=1)
        print(f"[{datetime.now()}] - Mejor Kappa: {study.best_value:.4f}")
        print(f"[{datetime.now()}] - Mejores parámetros: {study.best_params}")
        
        # Cargar test
        X_test, y_test = load_sparse_split(dataset_dir, "test")
        
        # Entrenar modelo final con los mejores parámetros en train+val
        X_train_val = vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        
        best_dt = DecisionTreeClassifier(**study.best_params, random_state=SEED)
        best_dt.fit(X_train_val, y_train_val)
        
        # Evaluación en test
        preds_test = best_dt.predict(X_test)
        test_acc = accuracy_score(y_test, preds_test)
        test_kappa = cohen_kappa_score(y_test, preds_test)
        
        print(f"[{datetime.now()}] - Test Accuracy: {test_acc:.4f}")
        print(f"[{datetime.now()}] - Test Kappa: {test_kappa:.4f}")
        
        # Guardar modelo
        model_dir = f"./models/decision_tree/{study_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"model_{study_name}.pkl")
        joblib.dump(best_dt, model_path)
        print(f"[{datetime.now()}] - Modelo guardado en {model_path}")
        
        # Guardar resultados
        results = {
            'best_params': study.best_params,
            'train_kappa': study.best_value,
            'test_accuracy': test_acc,
            'test_kappa': test_kappa,
        }
        
        results_path = os.path.join(model_dir, "results.pkl")
        joblib.dump(results, results_path)
        print(f"[{datetime.now()}] - Resultados guardados")
        
        return best_dt, results

    except Exception as e:
        print(f"[{datetime.now()}] - Error: {str(e)}")
        traceback.print_exc()
        return None, None
