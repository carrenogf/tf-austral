import os
import pandas as pd
from datetime import datetime
import traceback
import warnings
import optuna
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score
import joblib
from preprocessing import (
    ClassToInt,
    imputarNA,
    aplicar_ohe_splits
)

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 100
SEED = 12345

# Columnas necesarias (sin procesamiento de texto)
REQUIRED_COLS = [
    "tipo_comp", "nro_cuenta", "nro_entidad", "desc_entidad", "tipo_pres",
    "tipo_reg", "clase_reg", "cod", "fuente_fin", "descripcion", "tipo_cta",
    "cod_bco", "Class"
]

# Columnas para OneHotEncoding
OHE_COLS = ['tipo_comp', 'tipo_reg', 'clase_reg', 'tipo_cta', 'tipo_pres', 'desc_entidad', 'cod_bco', 'cod']


def load_and_preprocess(dataset_dir="./datasets"):
    """
    Carga los CSV directamente y aplica el mismo preprocesamiento que el pipeline sparse.
    Retorna X_train, y_train, X_val, y_val, X_test, y_test listos para usar.
    """
    try:
        dataset_dir = os.path.abspath(dataset_dir)
        
        print(f"[{datetime.now()}] - Cargando CSV desde {dataset_dir}")
        
        # Cargar los tres datasets
        dfs = {
            "train": pd.read_csv(os.path.join(dataset_dir, "2022-2023-2024.csv")),
            "val": pd.read_csv(os.path.join(dataset_dir, "2025-1.csv")),
            "test": pd.read_csv(os.path.join(dataset_dir, "2025-2.csv")),
        }
        
        print(f"[{datetime.now()}] - Train: {len(dfs['train'])} registros")
        print(f"[{datetime.now()}] - Val: {len(dfs['val'])} registros")
        print(f"[{datetime.now()}] - Test: {len(dfs['test'])} registros")
        
        # Seleccionar columnas necesarias
        print(f"[{datetime.now()}] - Seleccionando columnas requeridas...")
        for split in dfs:
            dfs[split] = dfs[split][REQUIRED_COLS]
        
        # Imputar NA's
        print(f"[{datetime.now()}] - Imputando valores NA...")
        for split in dfs:
            dfs[split] = imputarNA(dfs[split])
        
        # Convertir Class a entero usando el mapeo del train
        print(f"[{datetime.now()}] - Mapeando clases...")
        dfs["train"], class_mapping = ClassToInt(dfs["train"], class_mapping=None)
        dfs["val"], _ = ClassToInt(dfs["val"], class_mapping=class_mapping)
        dfs["test"], _ = ClassToInt(dfs["test"], class_mapping=class_mapping)
        print(f"[{datetime.now()}] - Mapeo de clases: {class_mapping}")
        
        # Descartar columna de descripción (no se usa procesamiento de texto)
        print(f"[{datetime.now()}] - Descartando columna 'descripcion' (sin procesamiento de texto)...")
        for split in dfs:
            dfs[split] = dfs[split].drop("descripcion", axis=1)
        
        # OneHotEncoding usando columnas definidas por train
        print(f"[{datetime.now()}] - Aplicando OneHotEncoding...")
        dfs["train"], dfs["val"], dfs["test"] = aplicar_ohe_splits(
            dfs["train"], dfs["val"], dfs["test"],
            columnas=OHE_COLS
        )
        
        # Separar features y target (la columna se llama "target" después de ClassToInt)
        X_train = dfs["train"].drop("target", axis=1)
        y_train = dfs["train"]["target"]
        
        X_val = dfs["val"].drop("target", axis=1)
        y_val = dfs["val"]["target"]
        
        X_test = dfs["test"].drop("target", axis=1)
        y_test = dfs["test"]["target"]
        
        print(f"[{datetime.now()}] - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except Exception as e:
        print(f"[{datetime.now()}] - Error en carga/preprocesamiento: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None, None


def modelo_completo(trials, study_name, dataset_dir="./datasets"):
    """
    Árbol de Decisiones usando los CSV directos (sin formato sparse).
    Optimiza contra val y reserva test para etapa posterior.
    """
    try:
        # Cargar y preprocesar datos
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess(dataset_dir)
        
        if X_train is None:
            print(f"[{datetime.now()}] - Error: No se pudieron cargar los datos")
            return None, None

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
            study = optuna.load_study(study_name=f"dt_base_{study_name}", storage=BBDD)
            print(f"[{datetime.now()}] - Estudio 'dt_base_{study_name}' cargado.")
        except:
            study = optuna.create_study(study_name=f"dt_base_{study_name}", storage=BBDD,
                                        direction='maximize')

        # Ejecutar optimización
        print(f"[{datetime.now()}] - Iniciando optimización con {trials} trials...")
        study.optimize(cv_es_dt_objective, n_trials=trials, n_jobs=1)
        print(f"[{datetime.now()}] - Mejor Kappa: {study.best_value:.4f}")
        print(f"[{datetime.now()}] - Mejores parámetros: {study.best_params}")
        
        # Entrenar modelo final con los mejores parámetros en train+val
        X_train_val = pd.concat([X_train, X_val], ignore_index=True)
        y_train_val = pd.concat([y_train, y_val], ignore_index=True)
        
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
        
        model_path = os.path.join(model_dir, f"model_base_{study_name}.pkl")
        joblib.dump(best_dt, model_path)
        print(f"[{datetime.now()}] - Modelo guardado en {model_path}")
        
        # Guardar resultados
        results = {
            'best_params': study.best_params,
            'train_kappa': study.best_value,
            'test_accuracy': test_acc,
            'test_kappa': test_kappa,
        }
        
        results_path = os.path.join(model_dir, "results_base.pkl")
        joblib.dump(results, results_path)
        print(f"[{datetime.now()}] - Resultados guardados")
        
        return best_dt, results

    except Exception as e:
        print(f"[{datetime.now()}] - Error: {str(e)}")
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Decision Tree Base - Entrenamiento con CSV sin preprocesamiento sparse")
    print("="*70 + "\n")
    
    model, results = modelo_completo(
        trials=TRIALS,
        study_name="base_model"
    )
    
    if model is not None and results is not None:
        print(f"\n[{datetime.now()}] - ¡Entrenamiento completado exitosamente!")
        print(f"Resultados finales:")
        print(f"  - Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  - Test Kappa: {results['test_kappa']:.4f}")
    else:
        print(f"\n[{datetime.now()}] - Error: El entrenamiento falló")
