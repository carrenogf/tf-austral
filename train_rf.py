import pandas as pd
import numpy as np
import joblib
import json
import os
import re
from datetime import datetime
def modelo_completo(trials, study_name, dataset_dir="./datasets"):
    """
    RandomForest con pesos + tf-idf. Optimiza contra df_val y guarda el pipeline entrenado con train+val.
    """
    try:
        dataset_dir = os.path.abspath(dataset_dir)
        train_path = os.path.join(dataset_dir, "df_final_train.csv")
        val_path = os.path.join(dataset_dir, "df_final_val.csv")

        print(f"[{datetime.now()}] - Leyendo train desde {train_path}")
        df_train = pd.read_csv(train_path, sep=';')
        print(f"[{datetime.now()}] - Leyendo val desde {val_path}")
        df_val = pd.read_csv(val_path, sep=';')

        # columnas 
        numeric_columns = get_numeric_columns(df_train)
        text_colummns = ["texto_limpio"]
        pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
        numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
        final_columns = numeric_columns + text_colummns + pesos_columns       
        df_train['texto_limpio'] = df_train['texto_limpio'].fillna('')
        df_val['texto_limpio'] = df_val['texto_limpio'].fillna('')

        # Preparar los datos
        X_train = df_train[final_columns]
        y_train = df_train["target"]
        X_val = df_val[final_columns]
        y_val = df_val["target"]

        # Definir los transformadores para el pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_columns),
                ('pesos', 'passthrough', pesos_columns),
                ('text', TfidfVectorizer(max_features=10000), "texto_limpio")
            ],
            remainder='drop'
        )

        def cv_es_rf_objective(trial):

            rfc_params = {      
                                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                                'max_depth': trial.suggest_int('max_depth', 3, 15),
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                                } 

            rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', rfc_model)
            ])

            pipeline.fit(X_train, y_train)
            preds_val = pipeline.predict(X_val)
            kappa = cohen_kappa_score(y_val, preds_val)
            trial.set_user_attr("val_accuracy", accuracy_score(y_val, preds_val))
            return kappa
  

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"randomforest_{study_name}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rf_objective, n_trials=trials)
        
        # guardamos mejor modelo
        print(f"[{datetime.now()}] - Mejores hiperparámetros: {study.best_params}\n")
        
        best_model = RandomForestClassifier(**study.best_params, random_state=SEED)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', best_model)
        ])

        # Entrenamos con train+val para maximizar datos antes de test
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)

        print(f"[{datetime.now()}] - Entrenando modelo con los mejores hiperparametros.. \n")
        pipeline.fit(X_full, y_full)

        model_dir = os.path.join("models", "randomforest", study_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'model_{study_name}.pkl')
        joblib.dump(pipeline, model_path)
        print(f"[{datetime.now()}] - Se ha guardado el modelo en {model_path} \n")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Se produjo un error: {e}")
        print(f"Detalles del error:\n{tb}")
        
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
        
        # kappa
        kappa_scorer = make_scorer(cohen_kappa_score)
        
        def cv_es_rf_objective(trial):

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

            # Crear el modelo RandomForestClassifier con los parámetros sugeridos
            rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)

            # Realizar validación cruzada usando Kappa como métrica
            kappa = cross_val_score(rfc_model, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
            
            return kappa

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"randomforest_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rf_objective, n_trials=TRIALS)
        
        
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

        # kappa
        kappa_scorer = make_scorer(cohen_kappa_score)
        
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

        def cv_es_rf_objective(trial):

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

            # Crear el modelo RandomForestClassifier con los parámetros sugeridos
            rfc_model = RandomForestClassifier(**rfc_params, random_state=SEED)
            
            # Crear pipeline completo
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', rfc_model)
            ])
            
            # Realizar validación cruzada usando Kappa como métrica
            kappa = cross_val_score(pipeline, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
            
            return kappa
  

        #Genero estudio
        study = optuna.create_study(direction='maximize', 
                                        storage=BBDD,  # Specify the storage URL here.
                                        study_name=f"randomforest_{STUDY_NAME}",
                                        load_if_exists=True)
            
        #Corro la optimizacion
        study.optimize(cv_es_rf_objective, n_trials=TRIALS)
        
        
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
    
    # kappa
    kappa_scorer = make_scorer(cohen_kappa_score)

    def cv_es_rf_objective(trial):
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
        
        # Realizar validación cruzada usando Kappa como métrica
        kappa = cross_val_score(rfc_model, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
        
        return kappa

        

    #Genero estudio
    study = optuna.create_study(direction='maximize', 
                                    storage=BBDD,  # Specify the storage URL here.
                                    study_name=f"randomforest_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rf_objective, n_trials=TRIALS)

    
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
    def cv_es_rf_objective(trial):

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
                                    study_name=f"randomforest_{STUDY_NAME}",
                                    load_if_exists=True)
        
    #Corro la optimizacion
    study.optimize(cv_es_rf_objective, n_trials=TRIALS)

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


