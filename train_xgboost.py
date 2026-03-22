import os
import pandas as pd
from datetime import datetime
import traceback
import warnings
import optuna
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from xgboost import XGBClassifier


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
        train_path = os.path.join(dataset_dir, "df_final_train.csv")
        val_path = os.path.join(dataset_dir, "df_final_val.csv")

        print(f"[{datetime.now()}] - Leyendo train desde {train_path}")
        df_train = pd.read_csv(train_path, sep=';')
        print(f"[{datetime.now()}] - Leyendo val desde {val_path}")
        df_val = pd.read_csv(val_path, sep=';')

        # columnas
        numeric_columns = get_numeric_columns(df_train)
        text_columns = ["texto_limpio"]
        pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
        numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
        final_columns = numeric_columns + text_columns + pesos_columns

        df_train['texto_limpio'] = df_train['texto_limpio'].fillna('')
        df_val['texto_limpio'] = df_val['texto_limpio'].fillna('')

        # Preparar los datos
        X_train = df_train[final_columns]
        y_train = df_train["target"]
        X_val = df_val[final_columns]
        y_val = df_val["target"]

        n_classes = int(y_train.nunique())

        # Definir los transformadores para el pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_columns),
                ('pesos', 'passthrough', pesos_columns),
                ('text', TfidfVectorizer(max_features=10000), "texto_limpio")
            ],
            remainder='drop'
        )

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

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', xgb_model)
            ])

            pipeline.fit(X_train, y_train)
            preds_val = pipeline.predict(X_val)
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

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])

        # Entrenamos con train+val para usar el mayor volumen posible antes de evaluar en test
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)

        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \\n")
        pipeline.fit(X_full, y_full)

        model_dir = os.path.join("models", "xgboost", study_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{study_name}.pkl")
        joblib.dump(pipeline, model_path)
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
