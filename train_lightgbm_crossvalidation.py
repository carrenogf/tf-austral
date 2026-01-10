import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import warnings
import lightgbm as lgb
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import importlib 
import archivos
import joblib

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 1000
SEED = 12345
TEST_SIZE = 0.2


def modelo_base():
    """
    Solo variables categoricas y numericas. No incluye pesos ni tf-idf.
    """
    # bdd
    importlib.reload(archivos)    
    STUDY_NAME = "modelo_base"
    
    # Leemos
    df = archivos.get_modelo_base()
    
    # Preparar los datos
    final_columns = [elemento for elemento in df.columns if elemento != 'target']
    X = df[final_columns]
    y = df["target"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    def cv_es_rfc_objective(trial):

        #Parametros para LightGBM
        param = {
            'objective': 'multiclass',
            'num_class': len(set(y)),  # Número de clases
            'metric': 'multi_logloss',  # Esto es solo para LightGBM; la métrica de optimización será accuracy
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'verbose': -1,
        }

        #Voy a generar estimaciones de los 5 modelos del CV sobre los datos test y los acumulo en la matriz scores_ensemble
        scores_ensemble = np.zeros((len(y_test),len(y_train.unique())))

        #Score del 5 fold CV inicializado en 0
        score_folds = 0

        #Numero de splits del CV
        n_splits = 5

        #Objeto para hacer el split estratificado de CV
        skf = StratifiedKFold(n_splits=n_splits)

        for i, (if_index, oof_index) in enumerate(skf.split(X_train, y_train)):
            
            # Dataset in fold (donde entreno)
            X_if, y_if = X_train.iloc[if_index], y_train.iloc[if_index]
            
            # Dataset Out of fold (donde mido la performance del CV)
            X_oof, y_oof = X_train.iloc[oof_index], y_train.iloc[oof_index]

            # Crear el dataset de LightGBM
            lgb_model = lgb.LGBMClassifier(**param,verbose_eval=False)
                        
            # Entrenar el modelo
            lgb_model.fit(X_if, y_if)
            
            # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
            scores_ensemble += lgb_model.predict_proba(X_test)
            
            # Score del fold (registros de dataset train que en este fold quedan out of fold)
            score_folds += cohen_kappa_score(y_oof, lgb_model.predict(X_oof), weights='quadratic') / n_splits
   

   
        #Determino score en conjunto de test y asocio como metrica adicional en optuna
        test_score = cohen_kappa_score(y_test,scores_ensemble.argmax(axis=1),weights = 'quadratic')
        trial.set_user_attr("test_score", test_score)

        #Devuelvo score del 5fold cv a optuna para que optimice en base a eso
        return(score_folds)

  

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"lightgbm_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rfc_objective, n_trials=TRIALS)
    
    
    # guardamos mejor modelo
    print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
    best_model = lgb.LGBMClassifier(**study.best_params, verbose_eval=False)
    print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f'models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
    print(f"[{datetime.now()}] - Se ha guardado el modelo en models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")
    
    
def modelo_text_mining():
    """
    Acá usamos los pesos calculados sobre el texto sin tf-idf
    """    
    try:
        # bdd
        STUDY_NAME = "modelo_text_mining"
        
        # Leemos
        df = archivos.modelo_text_mining()
        
        # Eliminamos columnas que nos nos sirven
        df.drop(columns=['texto_limpio'], inplace=True)
        
        # Preparar los datos
        final_columns = [elemento for elemento in df.columns if elemento != 'target']
        X = df[final_columns]
        y = df["target"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        def cv_es_rfc_objective(trial):

            #Parametros para LightGBM
            param = {
                'objective': 'multiclass',
                'num_class': len(set(y)),  # Número de clases
                'metric': 'multi_logloss',  # Esto es solo para LightGBM; la métrica de optimización será accuracy
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                'verbose': -1,
            }

            #Voy a generar estimaciones de los 5 modelos del CV sobre los datos test y los acumulo en la matriz scores_ensemble
            scores_ensemble = np.zeros((len(y_test),len(y_train.unique())))

            #Score del 5 fold CV inicializado en 0
            score_folds = 0

            #Numero de splits del CV
            n_splits = 5

            #Objeto para hacer el split estratificado de CV
            skf = StratifiedKFold(n_splits=n_splits)

            for i, (if_index, oof_index) in enumerate(skf.split(X_train, y_train)):
                
                # Dataset in fold (donde entreno)
                X_if, y_if = X_train.iloc[if_index], y_train.iloc[if_index]
                
                # Dataset Out of fold (donde mido la performance del CV)
                X_oof, y_oof = X_train.iloc[oof_index], y_train.iloc[oof_index]

                # Crear el modelo RandomForestClassifier con los parámetros sugeridos
                lgb_model = lgb.LGBMClassifier(**param,verbose_eval=False)
                            
                # Entrenar el modelo
                lgb_model.fit(X_if, y_if)
                
                # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
                scores_ensemble += lgb_model.predict_proba(X_test)
                
                # Score del fold (registros de dataset train que en este fold quedan out of fold)
                score_folds += cohen_kappa_score(y_oof, lgb_model.predict(X_oof), weights='quadratic') / n_splits
    

   
            #Determino score en conjunto de test y asocio como metrica adicional en optuna
            test_score = cohen_kappa_score(y_test,scores_ensemble.argmax(axis=1),weights = 'quadratic')
            trial.set_user_attr("test_score", test_score)

            #Devuelvo score del 5fold cv a optuna para que optimice en base a eso
            return(score_folds)

  

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"lightgbm_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rfc_objective, n_trials=TRIALS)
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        best_model = lgb.LGBMClassifier(**study.best_params, verbose_eval=False)
        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, f'models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
        print(f"[{datetime.now()}] - Se ha guardado el modelo en models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")






def modelo_completo():
    """
    Acá usamos pesos + tf-idf.
    """
    try:
        
        # bdd
        STUDY_NAME = "modelo_completo"
        
        # Leemos
        df = archivos.modelo_text_mining()
        
        # columnas 
        numeric_columns = get_numeric_columns(df)
        categorical_columns = get_categorical_columns(df, ['texto_limpio'])
        text_colummns = ["texto_limpio"]
        pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
        numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
        final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns       
        df['texto_limpio'] = df['texto_limpio'].fillna('')

 
        # Preparar los datos
        X = df[final_columns]
        y = df["target"]

    
        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)


        # Definir los transformadores para el pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_columns),  # No se aplica ningún preprocesamiento a las variables numéricas
                ('pesos', 'passthrough', pesos_columns),
                ('cat', 'passthrough', categorical_columns),
                ('text', TfidfVectorizer(max_features=10000), "texto_limpio")
            ],
            remainder='drop'
        )

        def cv_es_rfc_objective(trial):

            #Parametros para LightGBM
            param = {
                'objective': 'multiclass',
                'num_class': len(set(y)),  # Número de clases
                'metric': 'multi_logloss',  # Esto es solo para LightGBM; la métrica de optimización será accuracy
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                'verbose': -1,
            }

            #Voy a generar estimaciones de los 5 modelos del CV sobre los datos test y los acumulo en la matriz scores_ensemble
            scores_ensemble = np.zeros((len(y_test),len(y_train.unique())))

            #Score del 5 fold CV inicializado en 0
            score_folds = 0

            #Numero de splits del CV
            n_splits = 5

            #Objeto para hacer el split estratificado de CV
            skf = StratifiedKFold(n_splits=n_splits)

            for i, (if_index, oof_index) in enumerate(skf.split(X_train, y_train)):
                
                # Dataset in fold (donde entreno)
                X_if, y_if = X_train.iloc[if_index], y_train.iloc[if_index]
                
                # Dataset Out of fold (donde mido la performance del CV)
                X_oof, y_oof = X_train.iloc[oof_index], y_train.iloc[oof_index]

                # Crear el modelo RandomForestClassifier con los parámetros sugeridos
                lgb_model = lgb.LGBMClassifier(**param,verbose_eval=False)
                
                # Crear pipeline completo
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', lgb_model)
                ])
                
                # Entrenar el modelo
                pipeline.fit(X_if, y_if)
                
                # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
                scores_ensemble += pipeline.predict_proba(X_test)
                
                # Score del fold (registros de dataset train que en este fold quedan out of fold)
                score_folds += cohen_kappa_score(y_oof, pipeline.predict(X_oof), weights='quadratic') / n_splits

            
            #Determino score en conjunto de test y asocio como metrica adicional en optuna
            test_score = cohen_kappa_score(y_test,scores_ensemble.argmax(axis=1),weights = 'quadratic')
            trial.set_user_attr("test_score", test_score)

            #Devuelvo score del 5fold cv a optuna para que optimice en base a eso
            return(score_folds)

  

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"lightgbm_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rfc_objective, n_trials=TRIALS)
        
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        best_model = lgb.LGBMClassifier(**study.best_params, verbose_eval=False)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])
        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
        print(f"[{datetime.now()}] - Se ha guardado el modelo en models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")
    

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")
  


def modelo_tfidf():
    """
    Acá usamos solo tf-idf.
    """   
    # bdd
    STUDY_NAME = "modelo_tfidf"
    
    # Leemos
    df = archivos.modelo_text_mining()
    
    # columnas      
    df['texto_limpio'] = df['texto_limpio'].fillna('hola')
    
    # Preparar los datos
    y = df["target"]

    # Convertir texto a TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['texto_limpio'])

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    def cv_es_rfc_objective(trial):
       # Definir los hiperparámetros a optimizar
        param = {
            'objective': 'multiclass',
            'num_class': len(set(y)),  # Número de clases
            'metric': 'multi_logloss',  # Esto es solo para LightGBM; la métrica de optimización será accuracy
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'verbose': -1,
        }

        # Crear y entrenar el modelo
        lgb_model = lgb.LGBMClassifier(**param,verbose_eval=False)

        lgb_model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = lgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"lightgbm_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rfc_objective, n_trials=TRIALS)

    # Obtener los mejores hiperparámetros
    print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
    best_model = lgb.LGBMClassifier(**study.best_params, verbose_eval=False)
    
    # Entrenar el modelo
    print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
    best_model.fit(X_train, y_train)
    
    joblib.dump(best_model, f'models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl') 
    print(f"[{datetime.now()}] - Se ha guardado el modelo en models/lgbm/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")







    


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
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
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