import pandas as pd
import numpy as np
import joblib
import json
import os
import re
from datetime import datetime
import traceback
import warnings
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import optuna
from sklearn.metrics import make_scorer, cohen_kappa_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold 
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
from sklearn.ensemble import RandomForestClassifier
import importlib 
import archivos
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")
BBDD = "sqlite:///optuna.sqlite3"
TRIALS = 2
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
        rfc_params = {      
                            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                            'max_depth': trial.suggest_int('max_depth', 3, 15),
                            #'max_depth': trial.suggest_int('max_depth', 3, 4)
                            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
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
            rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)
                        
            # Entrenar el modelo
            rfc_model.fit(X_if, y_if)
            
            # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
            scores_ensemble += rfc_model.predict_proba(X_test)
            
            # Score del fold (registros de dataset train que en este fold quedan out of fold)
            score_folds += cohen_kappa_score(y_oof, rfc_model.predict(X_oof), weights='quadratic') / n_splits
   

   
        #Determino score en conjunto de test y asocio como metrica adicional en optuna
        test_score = cohen_kappa_score(y_test,scores_ensemble.argmax(axis=1),weights = 'quadratic')
        trial.set_user_attr("test_score", test_score)

        #Devuelvo score del 5fold cv a optuna para que optimice en base a eso
        return(score_folds)

  

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"randomforest1234_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rfc_objective, n_trials=TRIALS)
    
    # guardamos mejor modelo
    print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
    best_model = RandomForestClassifier(**study.best_params, random_state=SEED)
    print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f'models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
    print(f"[{datetime.now()}] - Se ha guardado el modelo en models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")
            
    
    
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
            rfc_params = {      
                                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                                'max_depth': trial.suggest_int('max_depth', 3, 15),
                                #'max_depth': trial.suggest_int('max_depth', 3, 4)
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
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
                rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)
                            
                # Entrenar el modelo
                rfc_model.fit(X_if, y_if)
                
                # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
                scores_ensemble += rfc_model.predict_proba(X_test)
                
                # Score del fold (registros de dataset train que en este fold quedan out of fold)
                score_folds += cohen_kappa_score(y_oof, rfc_model.predict(X_oof), weights='quadratic') / n_splits
    

   
            #Determino score en conjunto de test y asocio como metrica adicional en optuna
            test_score = cohen_kappa_score(y_test,scores_ensemble.argmax(axis=1),weights = 'quadratic')
            trial.set_user_attr("test_score", test_score)

            #Devuelvo score del 5fold cv a optuna para que optimice en base a eso
            return(score_folds)

  

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"randomforest1234_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rfc_objective, n_trials=TRIALS)
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        best_model = RandomForestClassifier(**study.best_params, random_state=SEED)
        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, f'models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
        print(f"[{datetime.now()}] - Se ha guardado el modelo en models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")
        
    
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
            rfc_params = {      
                                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                                'max_depth': trial.suggest_int('max_depth', 3, 15),
                                #'max_depth': trial.suggest_int('max_depth', 3, 4)
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
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
                rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)
                
                # Crear pipeline completo
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', rfc_model)
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
                                        study_name=f"randomforest1234_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rfc_objective, n_trials=TRIALS)
        
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        
        # Crear el modelo RandomForestClassifier con los parámetros sugeridos
        best_model = RandomForestClassifier(**study.best_params, random_state=SEED)
        
        # Crear pipeline completo
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])
        
        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
        print(f"[{datetime.now()}] - Se ha guardado el modelo en models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")

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
        rfc_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

        # Crear y entrenar el modelo
        rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)

        rfc_model.fit(X_train, y_train)

        # Evaluar el modelo
        y_pred = rfc_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"randomforest1234_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rfc_objective, n_trials=TRIALS)

    
    # guardamos mejor modelo
    print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
    best_model = RandomForestClassifier(**study.best_params, random_state=SEED)
    print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, f'models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl')           
    print(f"[{datetime.now()}] - Se ha guardado el modelo en models/randomforest/{STUDY_NAME}/model_{STUDY_NAME}.pkl \n")




def chatgpt():
    """
    Modelo recomendado por chatgpt para usa solo el tfidf y optuna
    """
    # bdd
    STUDY_NAME = "modelo_tfidf"
    
    # Leemos
    df = archivos.modelo_text_mining()
    
    # columnas      
    df['texto_limpio'] = df['texto_limpio'].fillna('')
    
    # Preparar los datos
    X = df["texto_limpio"]
    y = df["target"]

    #p split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Definir el transformador para convertir texto en TF-IDF
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=10000), 'texto_limpio')
        ],
        remainder='drop'
    )

    # Definir la función objetivo para Optuna
    def cv_es_rfc_objective(trial):

        # Parámetros a optimizar para RandomForest
        rfc_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

        # Inicializar la acumulación de scores para ensemble
        scores_ensemble = np.zeros((len(y_test), len(np.unique(y_train))))

        # Score del 5-fold CV inicializado en 0
        score_folds = 0

        # Estratificación para el K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5)

        for i, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):

            X_fold_train, X_fold_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_fold_train, y_fold_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            # Crear el modelo RandomForestClassifier con los parámetros sugeridos
            rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)

            # Crear pipeline completo
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', rfc_model)
            ])
            
            # Asegurarnos de pasar un DataFrame con la columna 'texto_limpio'
            X_fold_train_df = pd.DataFrame(X_fold_train, columns=['texto_limpio'])
            X_fold_valid_df = pd.DataFrame(X_fold_valid, columns=['texto_limpio'])

            # Entrenar el modelo
            pipeline.fit(X_fold_train_df, y_fold_train)

            # Predecir en validación
            pred_valid = pipeline.predict(X_fold_valid_df)

            # Acumular los scores (probabilidades) de cada clase para cada uno de los modelos que determino en los folds
            X_test_df = pd.DataFrame(X_test, columns=['texto_limpio'])
            scores_ensemble += pipeline.predict_proba(X_test_df)

            # Score del fold (registros de dataset train que en este fold quedan out of fold)
            score_folds += cohen_kappa_score(y_fold_valid, pred_valid, weights='quadratic') / 5

        # Determinar score en conjunto de test y asociarlo como métrica adicional en Optuna
        test_score = cohen_kappa_score(y_test, scores_ensemble.argmax(axis=1), weights='quadratic')
        trial.set_user_attr("test_score", test_score)

        return score_folds

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"randomforest1234_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rfc_objective, n_trials=TRIALS)

    # Obtener los mejores parámetros
    best_params = study.best_params
    print(f"Mejores parámetros: {best_params}")






    


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


