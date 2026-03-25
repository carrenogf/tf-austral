import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import warnings
import lightgbm as lgb
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from scipy.sparse import vstack
import importlib 
import archivos
import joblib
from sparse_dataset import load_sparse_split

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 1000
SEED = 12345
TEST_SIZE = 0.2

def modelo_completo(trials, study_name, dataset_dir="./datasets"):
    """
    Acá usamos pesos + tf-idf con los splits preprocesados (train/val/test).
    Optimizamos contra df_val y reservamos df_test para una etapa posterior.
    """
    try:
        dataset_dir = os.path.abspath(dataset_dir)
        print(f"[{datetime.now()}] - Leyendo train/val en formato sparse desde {dataset_dir}/final_sparse")
        X_train, y_train = load_sparse_split(dataset_dir, "train")
        X_val, y_val = load_sparse_split(dataset_dir, "val")

        def lgb_objective(trial):

            #Parametros para LightGBM
            param = {
                'objective': 'multiclass',
                'num_class': len(set(y_train.tolist())),
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
            }

            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            kappa = cohen_kappa_score(y_val, preds_val)
            trial.set_user_attr("val_accuracy", accuracy_score(y_val, preds_val))
            return kappa

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"LGBM_{study_name}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(lgb_objective, n_trials=trials)
        
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        model = lgb.LGBMClassifier(**study.best_params, verbose_eval=False)

        # Entrenamos con train+val para usar el mayor volumen posible antes de evaluar en test
        X_full = vstack([X_train, X_val], format='csr')
        y_full = np.concatenate([y_train, y_val])

        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        model.fit(X_full, y_full)

        model_dir = os.path.join("models", "lgbm", study_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'model_{study_name}.pkl')
        joblib.dump(model, model_path)
        print(f"[{datetime.now()}] - Se ha guardado el modelo en {model_path} \n")
    

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")
  




def get_numeric_columns(df):
  """
  Devuelve una lista de columnas numéricas en el DataFrame df.

  Parameters:
  df (pd.DataFrame): DataFrame del que obtener las columnas numéricas.

  Returns:
  list: Lista de nombres de columnas numéricas.
  """
  # Obtener las columnas numéricas
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  numeric_cols.remove("target")
  return numeric_cols



def get_categorical_columns(df, text_column):
    """
    Devuelve una lista de columnas categóricas en el DataFrame df,
    excluyendo la columna especificada (text_column).

    Parameters:
    df (pd.DataFrame): DataFrame del que obtener las columnas categóricas.
    text_column (str): Nombre de la columna de texto a excluir.

    Returns:
    list: Lista de nombres de columnas categóricas excluyendo text_column.
    """
    # Obtener las columnas categóricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Excluir la columna de texto
    for a in text_column:
        if a in categorical_cols:
            categorical_cols.remove(a)
    
    return categorical_cols




def model_lightgbm(study_name,ntrials):
  try:
    # leer datos 
    df = pd.read_csv(f"models/lgbm/{study_name}/df_train.csv")
    bbdd = "sqlite:///optuna.sqlite3"
    
    # convertir columnas a nro
    for col in df.columns:
      if df[col].dtype == 'object':
       df[col] = pd.to_numeric(df[col], errors='ignore')
    

    X = df.drop(columns=["target"])
    y = df["target"]
    
    
    SEED = 12345
    TEST_SIZE = 0.2
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

   
    kappa_scorer = make_scorer(cohen_kappa_score)
    def objective(trial):
        param = {
            'objective': 'multiclass',
            'num_class': len(set(y)),  # Número de clases
            'metric': 'multi_logloss',  # Esto es solo para LightGBM; la métrica de optimización será accuracy
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
        }
        
        # Crear el dataset de LightGBM
        model = lgb.LGBMClassifier(**param,verbose_eval=False)
        
        # Realizar validación cruzada usando Kappa como métrica
        kappa = cross_val_score(model, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
        
        return kappa


    # Crear un estudio y optimizar
    study = optuna.create_study(direction='maximize', 
                                storage=bbdd,  # Specify the storage URL here.
                                study_name=f"lightgbm_{study_name}",
                                load_if_exists=True)
    study.optimize(objective, n_trials=ntrials)
    
    
    
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")