import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import warnings
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    make_scorer, cohen_kappa_score, accuracy_score, 
    precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import vstack
import joblib
import json
from sparse_dataset import load_sparse_split

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 500
SEED = 12345

def walk_forward_validation(
    X_full, y_full, 
    initial_train_ratio=0.6, 
    step_size=0.1,
    n_splits=None,
    trials=TRIALS,
    study_name="walkforward"
):
    """
    Realiza validación walk forward con optimización de hiperparámetros.
    
    Parameters:
    -----------
    X_full : sparse matrix
        Características completas (ordenadas temporalmente)
    y_full : array
        Target completo (ordenado temporalmente)
    initial_train_ratio : float
        Proporción inicial de datos para entrenamiento (default: 0.6)
    step_size : float
        Proporción de datos a avanzar en cada paso (default: 0.1)
    n_splits : int
        Número de splits. Si es None, se calcula automáticamente basado en step_size
    trials : int
        Número de trials para Optuna
    study_name : str
        Nombre del estudio Optuna
    
    Returns:
    --------
    dict : Diccionario con resultados de cada fold y modelos
    """
    
    n_samples = X_full.shape[0]
    
    # Calcular splits si no se especifican
    if n_splits is None:
        n_splits = max(2, int((1 - initial_train_ratio) / step_size))
    
    initial_train_size = int(n_samples * initial_train_ratio)
    step_size_samples = int(n_samples * step_size)
    
    print(f"[{datetime.now()}] - Iniciando Walk Forward Validation")
    print(f"  - Total de muestras: {n_samples}")
    print(f"  - Tamaño inicial de entrenamiento: {initial_train_size} ({initial_train_ratio*100:.1f}%)")
    print(f"  - Paso de avance: {step_size_samples} muestras ({step_size*100:.1f}%)")
    print(f"  - Número de splits: {n_splits}\n")
    
    results = {
        'fold_results': [],
        'models': [],
        'best_hyperparams': None,
        'overall_metrics': {}
    }
    
    all_preds = []
    all_true = []
    best_score = -np.inf
    global_best_params = None
    
    # Walk Forward Loop
    for fold in range(n_splits):
        print(f"\n{'='*70}")
        print(f"[{datetime.now()}] - FOLD {fold + 1}/{n_splits}")
        print(f"{'='*70}")
        
        # Definir índices
        train_end = initial_train_size + (fold * step_size_samples)
        test_end = train_end + step_size_samples
        
        # Asegurar que no nos salimos de los datos
        if train_end >= n_samples or test_end > n_samples:
            print(f"[{datetime.now()}] - Se alcanzó el final de los datos. Deteniendo.")
            break
        
        # Dividir datos
        X_train_fold = X_full[:train_end]
        y_train_fold = y_full[:train_end]
        X_val_fold = X_full[train_end:test_end]
        y_val_fold = y_full[train_end:test_end]
        
        print(f"[{datetime.now()}] - Entrenamiento: índices [0:{train_end}] ({len(y_train_fold)} muestras)")
        print(f"[{datetime.now()}] - Validación: índices [{train_end}:{test_end}] ({len(y_val_fold)} muestras)")
        
        # Optimización de hiperparámetros para este fold
        def lgb_objective_fold(trial):
            param = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_train_fold)),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 512),
                'max_depth': trial.suggest_int('max_depth', -1, 30),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
                'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'verbose': -1,
                'random_state': SEED,
            }
            
            model = lgb.LGBMClassifier(**param)
            model.fit(X_train_fold, y_train_fold)
            preds_val = model.predict(X_val_fold)
            kappa = cohen_kappa_score(y_val_fold, preds_val)
            trial.set_user_attr("val_accuracy", accuracy_score(y_val_fold, preds_val))
            return kappa
        
        # Crear estudio
        study_fold_name = f"LGBM_{study_name}_fold{fold}"
        study = optuna.create_study(
            direction='maximize',
            storage=BBDD,
            study_name=study_fold_name,
            load_if_exists=True
        )
        
        print(f"[{datetime.now()}] - Optimizando hiperparámetros ({trials} trials)...")
        study.optimize(lgb_objective_fold, n_trials=trials, show_progress_bar=False)
        
        best_params_fold = study.best_params
        best_kappa_fold = study.best_value
        
        print(f"[{datetime.now()}] - Mejores parámetros en fold {fold + 1}:")
        print(f"  - Kappa: {best_kappa_fold:.4f}")
        for key, val in best_params_fold.items():
            print(f"  - {key}: {val}")
        
        # Entrenar modelo final con mejores parámetros
        model_fold = lgb.LGBMClassifier(**best_params_fold)
        model_fold.fit(X_train_fold, y_train_fold)
        
        # Evaluar en validación
        preds_val_fold = model_fold.predict(X_val_fold)
        probs_val_fold = model_fold.predict_proba(X_val_fold)
        
        # Calcular métricas
        accuracy_fold = accuracy_score(y_val_fold, preds_val_fold)
        kappa_fold = cohen_kappa_score(y_val_fold, preds_val_fold)
        precision_fold = precision_score(y_val_fold, preds_val_fold, average='weighted', zero_division=0)
        recall_fold = recall_score(y_val_fold, preds_val_fold, average='weighted', zero_division=0)
        f1_fold = f1_score(y_val_fold, preds_val_fold, average='weighted', zero_division=0)
        
        fold_result = {
            'fold': fold,
            'train_size': len(y_train_fold),
            'val_size': len(y_val_fold),
            'accuracy': accuracy_fold,
            'kappa': kappa_fold,
            'precision': precision_fold,
            'recall': recall_fold,
            'f1': f1_fold,
            'best_params': best_params_fold,
        }
        
        results['fold_results'].append(fold_result)
        results['models'].append(model_fold)
        
        all_preds.extend(preds_val_fold.tolist())
        all_true.extend(y_val_fold.tolist())
        
        # Guardar el mejor modelo
        if kappa_fold > best_score:
            best_score = kappa_fold
            global_best_params = best_params_fold
            results['best_hyperparams'] = best_params_fold
            best_fold = fold
        
        print(f"\n[{datetime.now()}] - Métricas del fold {fold + 1}:")
        print(f"  - Accuracy: {accuracy_fold:.4f}")
        print(f"  - Kappa: {kappa_fold:.4f}")
        print(f"  - Precision (weighted): {precision_fold:.4f}")
        print(f"  - Recall (weighted): {recall_fold:.4f}")
        print(f"  - F1 (weighted): {f1_fold:.4f}")
    
    # Calcular métricas globales
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    overall_accuracy = accuracy_score(all_true, all_preds)
    overall_kappa = cohen_kappa_score(all_true, all_preds)
    overall_precision = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    overall_recall = recall_score(all_true, all_preds, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0)
    
    results['overall_metrics'] = {
        'accuracy': overall_accuracy,
        'kappa': overall_kappa,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
    }
    
    print(f"\n{'='*70}")
    print(f"[{datetime.now()}] - MÉTRICAS GLOBALES (todos los folds)")
    print(f"{'='*70}")
    print(f"  - Accuracy: {overall_accuracy:.4f}")
    print(f"  - Kappa: {overall_kappa:.4f}")
    print(f"  - Precision (weighted): {overall_precision:.4f}")
    print(f"  - Recall (weighted): {overall_recall:.4f}")
    print(f"  - F1 (weighted): {overall_f1:.4f}")
    print(f"  - Mejor fold: {best_fold} (Kappa: {best_score:.4f})")
    
    return results


def save_walkforward_results(results, study_name, dataset_dir="./datasets"):
    """
    Guarda los resultados y modelos de walk forward validation.
    """
    try:
        model_dir = os.path.join("models", "lgbm", f"{study_name}_walkforward")
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar resultados detallados
        results_summary = {
            'fold_results': results['fold_results'],
            'overall_metrics': results['overall_metrics'],
            'best_hyperparams': results['best_hyperparams'],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(model_dir, 'walkforward_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"[{datetime.now()}] - Resultados guardados en {results_path}")
        
        # Guardar cada modelo
        for fold, model in enumerate(results['models']):
            model_path = os.path.join(model_dir, f'model_fold_{fold}.pkl')
            joblib.dump(model, model_path)
        
        print(f"[{datetime.now()}] - {len(results['models'])} modelos guardados en {model_dir}")
        
        # Guardar hiperparámetros del mejor modelo
        params_path = os.path.join(model_dir, 'best_hyperparams.json')
        with open(params_path, 'w') as f:
            json.dump(results['best_hyperparams'], f, indent=4)
        
        return model_dir
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error guardando resultados: {e}")
        print(f"Detalles: {tb}")
        return None


def main():
    """
    Función principal que ejecuta el walk forward validation completo.
    """
    try:
        dataset_dir = os.path.abspath("./datasets")
        study_name = "sparse_walkforward"
        
        print(f"[{datetime.now()}] - Cargando datos sparse desde {dataset_dir}/final_sparse")
        X_train, y_train = load_sparse_split(dataset_dir, "train")
        X_val, y_val = load_sparse_split(dataset_dir, "val")
        X_test, y_test = load_sparse_split(dataset_dir, "test")
        
        # Combinar train + val + test para walk forward
        X_full = vstack([X_train, X_val, X_test], format='csr')
        y_full = np.concatenate([y_train, y_val, y_test])
        
        print(f"[{datetime.now()}] - Datos combinados: {X_full.shape[0]} muestras, {X_full.shape[1]} features")
        print(f"[{datetime.now()}] - Clases: {np.unique(y_full)}\n")
        
        # Ejecutar walk forward
        results = walk_forward_validation(
            X_full=X_full,
            y_full=y_full,
            initial_train_ratio=0.6,
            step_size=0.1,
            n_splits=None,
            trials=TRIALS,
            study_name=study_name
        )
        
        # Guardar resultados
        model_dir = save_walkforward_results(results, study_name, dataset_dir)
        
        if model_dir:
            print(f"\n[{datetime.now()}] - Walk Forward Validation completado exitosamente!")
            print(f"[{datetime.now()}] - Resultados guardados en: {model_dir}")
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error en main: {e}")
        print(f"Detalles: {tb}")


if __name__ == "__main__":
    main()
