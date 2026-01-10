import pandas as pd
import numpy as np
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
from sklearn.metrics import make_scorer, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
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
from sklearn.metrics import accuracy_score
from optuna.integration import KerasPruningCallback  # Importación correcta
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
#import autokeras as ak
try:
  STOPWORDS = set(stopwords.words('spanish'))
  nlp = spacy.load("es_core_news_sm")
except Exception as e:
  print(e)
  # !pip install spacy
  # !python -m spacy download es_core_news_sm
  # STOPWORDS = set(stopwords.words('spanish'))
  # nlp = spacy.load("es_core_news_sm")

from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

def train_red_neuronal(path_df, study_name, ntrials, flag_generar_archivo=False):
  """
  Entrenamamos un modelo de red neuronal secuencial con Keras.
  """
  try:
    # leer datos
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/rn", exist_ok=True)
    os.makedirs(f"models/rn/{study_name}", exist_ok=True)    
    bbdd = "sqlite:///optuna.sqlite3"
    
    if( flag_generar_archivo ):
      # df_rnn = preprocessing + feature engineering ------------------------------------
      print(f"{datetime.now()} - Inicio preprocesamiento\n")
      df = preprocessing(path_df)
      print(f"{datetime.now()} - Inicio feature engineering\n")
      df = featureEngineering(df)
      df["texto_limpio"] = df["texto_limpio"].fillna('')
      df.to_csv(f"../datasets/df_rnn.csv", index=False, sep=";")
    
    
    
    # abrimos directamente el archiv
    df = pd.read_csv(path_df, sep=";")
    
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df, ['Descripcion', 'texto_limpio'])
    text_colummns = ["texto_limpio"]
    pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
    numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
    final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
    df["texto_limpio"] = df["texto_limpio"].fillna('')
    
    # Preparar los datos
    X = df[final_columns]
    y = df["target"].values

    SEED = 12345
    TEST_SIZE = 0.2


    # Definir los transformadores para el pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_columns),  # No se aplica ningún preprocesamiento a las variables numéricas
            ('pesos', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), pesos_columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_columns),
            ('text', TfidfVectorizer(max_features=10000), "texto_limpio")
        ],
        remainder='drop'
    )
    
    
    
    # Función para construir el modelo Keras
    def build_nn(input_dim):
        model = Sequential([
            Dense(1000, input_shape=(input_dim,)),
            Activation('relu'),
            Dropout(0.5),
            Dense(500),
            Activation('relu'),
            Dropout(0.5),
            Dense(50),
            Activation('relu'),
            Dropout(0.5),
            Dense(27),  # Ajustar según el número de clases en la salida
            Activation('softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Adaptar el modelo Keras para usarlo en Scikit-learn
    output_dim = len(np.unique(y))  # Número de clases
    nn_model = KerasClassifier(
        model=build_nn,
        # output_dim=output_dim,
        epochs=20,
        batch_size=64,
        verbose=1
    )

    # Pipeline de preprocesamiento y modelo
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', nn_model)
    ])
    

   

    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

    # Convertir las etiquetas a formato one-hot si es multiclase
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Obtener el input_dim después de transformar los datos
    X_train_transformed = model_pipeline.named_steps['preprocessor'].fit_transform(X_train)
    input_dim = X_train_transformed.shape[1]

    # Redefinir el pipeline con el input_dim correcto
    model_pipeline.set_params(model__model=build_nn(input_dim))

    # Entrenar el modelo
    model_pipeline.fit(X_train, y_train)
    
    kappa_scorer = make_scorer(cohen_kappa_score)    
    def objective(trial):
      # Probar distintos hiperparámetros
      n_layers = trial.suggest_int('n_layers', 2, 4)
      units = [trial.suggest_int(f'units_l{i}', 64, 512) for i in range(n_layers)]
      dropout = trial.suggest_float('dropout', 0.2, 0.5)
      learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
      batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
      epochs = trial.suggest_int('epochs', 10, 50)
      
      def build_model(input_dim):
          model = Sequential()
          for i in range(n_layers):
              model.add(Dense(units[i], input_shape=(input_dim,), activation='relu'))
              model.add(Dropout(dropout))
          model.add(Dense(27, activation='softmax'))
          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['kappa'])
          return model
      
      model_pipeline.set_params(model__model=build_model(input_dim))
      model_pipeline.set_params(model__batch_size=batch_size, model__epochs=epochs)
      
      # model_pipeline.fit(X_train, y_train)
      # accuracy = model_pipeline.score(X_test, y_test)
      # return accuracy
      # Realizar validación cruzada usando Kappa como métrica
      kappa = cross_val_score(model_pipeline, X_train, y_train, cv=3, scoring=kappa_scorer).mean()
        
      return kappa
    
    
    # Crear un estudio y optimizar
    print(f"{datetime.now()} - Inicio optimizacion de hiperparametros \n")
    study = optuna.create_study(direction='maximize', 
                                storage=bbdd,  # Specify the storage URL here.
                                study_name=f"rnborraadsar_{study_name}",
                                load_if_exists=True)
    study.optimize(objective, n_trials=ntrials)

 
    # Imprimir los mejores hiperparámetros encontrados
    print('Mejores hiperparámetros:', study.best_params)
    print('Mejor valor de Kappa:', study.best_value)
        
    
    
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")

  
def train_red_neuronal_sin_optuna(path_df, study_name):
  """
  """
  # abrimos directamente el archiv
  df = pd.read_csv(path_df, sep=";")
  
  
  numeric_columns = get_numeric_columns(df)
  categorical_columns = get_categorical_columns(df, ['Descripcion', 'texto_limpio'])
  text_colummns = ["texto_limpio"]
  pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
  numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
  final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
  df["texto_limpio"] = df["texto_limpio"].fillna('')
  X_text = df['texto_limpio'].values  # Columna de texto
  
  # Preprocesamiento de datos numéricos y booleanos
  scaler = StandardScaler()
  X_pesos_scaled = scaler.fit_transform(df[pesos_columns])
  

  # Preprocesamiento de texto usando TF-IDF
  vectorizer = TfidfVectorizer(max_features=10000)
  X_text_tfidf = vectorizer.fit_transform(X_text).toarray()

  # Combinar datos numéricos y texto
  X_combined = np.hstack((df[numeric_columns].values, X_pesos_scaled, X_text_tfidf))
  
    # Preparar los datos
  X = df[final_columns]
  y = df["target"].values

  SEED = 12345
  TEST_SIZE = 0.2
  batch_size = 64
  nb_epochs = 20
  
    # División en conjunto de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=TEST_SIZE, random_state=SEED)

  # Convertir las etiquetas a formato one-hot si es multiclase
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  # Definir la red neuronal
  model = Sequential()

  model.add(Dense(1000, input_shape=(X_combined.shape[1],)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(500))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(27))  # Número de clases en la salida
  model.add(Activation('softmax'))

  # Compilar el modelo
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Entrenar el modelo
  #model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test), verbose=1)
  model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test), verbose=1)
   
  # Hacer predicciones
  y_train_predclass = model.predict(X_train, batch_size=batch_size)
  y_test_predclass = model.predict(X_test, batch_size=batch_size)

  # Convertir las etiquetas y las predicciones a formato ordinal
  y_train_labels = np.argmax(y_train, axis=1)
  y_test_labels = np.argmax(y_test, axis=1)
  y_train_pred_labels = np.argmax(y_train_predclass, axis=1)
  y_test_pred_labels = np.argmax(y_test_predclass, axis=1)
  
  

  # Calcular la precisión y generar el informe de clasificación
  print("Train accuracy: {}".format(round(accuracy_score(y_train_labels, y_train_pred_labels), 3)))
  print("Test accuracy: {}".format(round(accuracy_score(y_test_labels, y_test_pred_labels), 3)))
  print("Train kappa: {}".format(round(cohen_kappa_score(y_train_labels, y_train_pred_labels), 3)))
  print("Test kappa: {}".format(round(cohen_kappa_score(y_test_labels, y_test_pred_labels), 3)))
  print("\nTest Classification Report\n")
  print(classification_report(y_test_labels, y_test_pred_labels))
  


  

def model_rnn_optuna(path_df, study_name, ntrials):
  df = pd.read_csv(path_df, sep=";")
  bbdd = "sqlite:///optuna.sqlite3"
  numeric_columns = get_numeric_columns(df)
  categorical_columns = get_categorical_columns(df, ['Descripcion', 'texto_limpio'])
  text_colummns = ["texto_limpio"]
  pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
  numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
  final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
  df["texto_limpio"] = df["texto_limpio"].fillna('')
  X_text = df['texto_limpio'].values  # Columna de texto

  # Preprocesamiento de datos numéricos y booleanos
  scaler = StandardScaler()
  X_pesos_scaled = scaler.fit_transform(df[pesos_columns])

  # Preprocesamiento de texto usando TF-IDF
  vectorizer = TfidfVectorizer(max_features=10000)
  X_text_tfidf = vectorizer.fit_transform(X_text).toarray()

  # Combinar datos numéricos y texto
  X_combined = np.hstack((df[numeric_columns].values, X_pesos_scaled, X_text_tfidf))

  # Preparar los datos
  y = df["target"].values

  SEED = 12345
  TEST_SIZE = 0.2
  X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=TEST_SIZE, random_state=SEED)

  # Convertir las etiquetas a formato one-hot si es multiclase
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  # Función objetivo para Optuna
  def objective(trial):
      # Hiperparámetros a optimizar
      n_layers = trial.suggest_int('n_layers', 2, 4)
      units = [trial.suggest_int(f'units_l{i}', 64, 512) for i in range(n_layers)]
      dropout = trial.suggest_float('dropout', 0.2, 0.5)
      learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
      #batch_size = trial.suggest_categorical('batch_size', [32,64])
      nb_epochs = trial.suggest_int('epochs', 20, 50)
      

      # Definir la red neuronal
      model = Sequential()
      model.add(Dense(units[0], input_shape=(X_train.shape[1],)))
      model.add(Activation('relu'))
      model.add(Dropout(dropout))

      for i in range(1, n_layers):
          model.add(Dense(units[i]))
          model.add(Activation('relu'))
          model.add(Dropout(dropout))

      model.add(Dense(27))  # Número de clases en la salida
      model.add(Activation('softmax'))

      # Compilar el modelo
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      # Callback para permitir el pruning (opcional)
      callbacks = [KerasPruningCallback(trial, "val_accuracy")]

      # Entrenar el modelo
      model.fit(X_train, y_train, validation_data=(X_test, y_test),
                          batch_size=64, epochs=nb_epochs, verbose=0,
                          callbacks=callbacks)

      # Evaluar usando Kappa
      y_pred = np.argmax(model.predict(X_test), axis=1)
      y_true = np.argmax(y_test, axis=1)
      kappa_score = cohen_kappa_score(y_true, y_pred)

      return kappa_score

  # Crear un estudio y optimizar
  print(f"{datetime.now()} - Inicio optimizacion de hiperparametros \n")
  study = optuna.create_study(direction='maximize', 
                              storage=bbdd,  # Specify the storage URL here.
                              study_name=f"rnsborraadsar_{study_name}",
                              load_if_exists=True)
  study.optimize(objective, n_trials=ntrials)


  # Imprimir los mejores hiperparámetros encontrados
  print('Mejores hiperparámetros:', study.best_params)
  print('Mejor valor de Kappa:', study.best_value)
  

def model_rnn_optuna_sin_texto(path_df, study_name, ntrials):
  df = pd.read_csv(path_df, sep=";")
  bbdd = "sqlite:///optuna.sqlite3"
  numeric_columns = get_numeric_columns(df)
  categorical_columns = get_categorical_columns(df, ['Descripcion', 'texto_limpio'])
  
  pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
  numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
  final_columns = numeric_columns + categorical_columns  + pesos_columns
  
  

  # Preprocesamiento de datos numéricos y booleanos
  scaler = StandardScaler()
  X_pesos_scaled = scaler.fit_transform(df[pesos_columns])

  # Aplicar OneHotEncoding a las columnas categóricas
  onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
  X_categorical_encoded = onehot_encoder.fit_transform(df[categorical_columns])

  # Convertir el resultado en un DataFrame
  encoded_columns = onehot_encoder.get_feature_names_out(categorical_columns)
  X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=encoded_columns)

  # Reiniciar los índices para que coincidan con el DataFrame original
  X_categorical_encoded_df.index = df.index

  # Eliminar las columnas categóricas originales del DataFrame
  df = df.drop(columns=categorical_columns)

  # Combinar el DataFrame original con las columnas codificadas
  # df = pd.concat([df, X_categorical_encoded_df], axis=1)

  # Combinar datos numéricos y texto
  X_combined = np.hstack((df[numeric_columns].values, X_pesos_scaled, X_categorical_encoded_df))

  # Preparar los datos
  y = df["target"].values

  SEED = 12345
  TEST_SIZE = 0.2
  X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=TEST_SIZE, random_state=SEED)

  # Convertir las etiquetas a formato one-hot si es multiclase
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  # Función objetivo para Optuna
  def objective(trial):
      # Hiperparámetros a optimizar
      n_layers = trial.suggest_int('n_layers', 2, 4)
      units = [trial.suggest_int(f'units_l{i}', 64, 128) for i in range(n_layers)]
      dropout = trial.suggest_float('dropout', 0.2, 0.5)
      learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
      #batch_size = trial.suggest_categorical('batch_size', [32,64])
      nb_epochs = trial.suggest_int('epochs', 20, 50)
      

      # Definir la red neuronal
      model = Sequential()
      model.add(Dense(units[0], input_shape=(X_train.shape[1],)))
      model.add(Activation('relu'))
      model.add(Dropout(dropout))

      for i in range(1, n_layers):
          model.add(Dense(units[i]))
          model.add(Activation('relu'))
          model.add(Dropout(dropout))

      model.add(Dense(27))  # Número de clases en la salida
      model.add(Activation('softmax'))

      # Compilar el modelo
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      # Callback para permitir el pruning (opcional)
      callbacks = [KerasPruningCallback(trial, "val_accuracy")]

      # Entrenar el modelo
      model.fit(X_train, y_train, validation_data=(X_test, y_test),
                          batch_size=64, epochs=nb_epochs, verbose=0,
                          callbacks=callbacks)

      # Evaluar usando Kappa
      y_pred = np.argmax(model.predict(X_test), axis=1)
      y_true = np.argmax(y_test, axis=1)
      kappa_score = cohen_kappa_score(y_true, y_pred)

      return kappa_score

  # Crear un estudio y optimizar
  print(f"{datetime.now()} - Inicio optimizacion de hiperparametros \n")
  study = optuna.create_study(direction='maximize', 
                              storage=bbdd,  # Specify the storage URL here.
                              study_name=f"rnsborraadsar_{study_name}",
                              load_if_exists=True)
  study.optimize(objective, n_trials=ntrials)


  # Imprimir los mejores hiperparámetros encontrados
  print('Mejores hiperparámetros:', study.best_params)
  print('Mejor valor de Kappa:', study.best_value)


def red_neuronal_basica_ivan(path_df, study, trials):
  """
  ESTE ANDA PAPA!!
  """
  df = pd.read_excel(path_df)
  print(f"{datetime.now()} - Inicio preprocessing \n")
  df.drop(columns=["DescCuenta","NTesoreria","DescTesoreria","DescEntidad","Beneficiario"], axis=1)
  df["Descripcion"] = df["Descripcion"].fillna("")
  df["ClaseReg"] = df["ClaseReg"].fillna("Indefinido")
  Class = list(df.Class.unique())
  clases = {val:Class.index(val) for val in Class}
  def get_class(val):
      return clases[val]
  df['target'] = df['Class'].apply(get_class)
  print(f"{datetime.now()} - Inicio feature engineering \n")
  df['Descripcion'] = df['Descripcion'].astype(str)   
  df['text_size'] = df['Descripcion'].str.len()
  df['text_words_count'] = df['Descripcion'].apply(lambda x: len(x.split()))
  print(f"{datetime.now()} - Calculamos pesos \n")
  dictOfWords = {}
  for target in df.target.unique():
    df_target = df[df["target"]==target]
    all_descriptions = ' '.join(df_target['Descripcion'].dropna())  # Concatenar todas las descripciones
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(all_descriptions)
    clean_text = []
    for token in doc:
        if (
            not token.is_stop  # No incluir stopwords
            and not token.is_punct  # No incluir puntuación
            and not token.like_num  # Opción para excluir números si prefieres solo palabras
        ):
            #clean_text.append(token.lemma_.lower())
            clean_text.append(str(token.lemma_).lower())
    # Filtrar stopwords y no palabras alfa
    freq_of_words = pd.Series(clean_text).value_counts()
    dic_words = freq_of_words.to_dict()
    dictOfWords[str(target)] = dic_words
    df[f'pesos_{target}'] = df['Descripcion'].apply(pesos,dic_words=dic_words)

  # Parámetros
  nb_classes = df['target'].nunique()  # Número de clases en la columna target
  max_words = 10000  # Número máximo de palabras a considerar
  batch_size = 64
  nb_epochs = 20
  print(f"{datetime.now()} - Tokenizamos \n")
  # Tokenizar el texto (convierte 'texto_limpio' de objeto a una matriz numérica)
  tokenizer = Tokenizer(num_words=max_words)
  tokenizer.fit_on_texts(df['Descripcion'].astype(str))  # Asegúrate de que sea tratado como cadena de texto
  X = tokenizer.texts_to_matrix(df['Descripcion'].astype(str), mode='tfidf')  # Convierte el texto a TF-IDF

  # Convertir la columna 'target' a one-hot encoding
  y = to_categorical(df['target'], nb_classes)

  SEED = 12345
  TEST = 0.2
  # Dividir el dataset en train y test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST, random_state=SEED)
  
  # Crear el modelo
  model = Sequential()
  model.add(Dense(1000, input_shape=(max_words,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(500))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nb_classes))
  model.add(Activation('softmax'))

  # Compilar el modelo
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(f"{datetime.now()} - Entrenamos modelo \n")
  # Entrenar el modelo
  model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test), verbose=1)

  # Hacer predicciones
  y_train_predclass = model.predict(X_train, batch_size=batch_size)
  y_test_predclass = model.predict(X_test, batch_size=batch_size)

  # Convertir las etiquetas y las predicciones a formato ordinal
  y_train_labels = np.argmax(y_train, axis=1)
  y_test_labels = np.argmax(y_test, axis=1)
  y_train_pred_labels = np.argmax(y_train_predclass, axis=1)
  y_test_pred_labels = np.argmax(y_test_predclass, axis=1)

  # Calcular la precisión y generar el informe de clasificación
  print("Train accuracy: {}".format(round(accuracy_score(y_train_labels, y_train_pred_labels), 3)))
  print("Test accuracy: {}".format(round(accuracy_score(y_test_labels, y_test_pred_labels), 3)))
  print("\nTest Classification Report\n")
  print(classification_report(y_test_labels, y_test_pred_labels))




def model_autokeras(path_df, study, trials, flag_file=False):
  """
  !pip install autokeras
  """
  
  if( flag_file ):
    print(f"{datetime.now()} - Inicio preprocesamiento\n")
    df = preprocessing(path_df)
    print(f"{datetime.now()} - Inicio feature engineering\n")
    df = featureEngineering(df)
    df["texto_limpio"] = df["texto_limpio"].fillna('')
    df.to_csv(f"../datasets/df_final.csv", index=False, sep=";")
    
  
  print(f"[{datetime.now()}] - Leemos el dataset \n")
  df = pd.read_csv("../datasets/df_final.csv", sep=";")
  
   
  numeric_columns = get_numeric_columns(df)
  categorical_columns = get_categorical_columns(df, ['Descripcion', 'texto_limpio'])
  text_colummns = ["texto_limpio"]
  pesos_columns = [col for col in numeric_columns if col.startswith('pesos_')]
  numeric_columns = [col for col in numeric_columns if not col.startswith('pesos_')]
  final_columns = numeric_columns + categorical_columns + text_colummns + pesos_columns
  df["texto_limpio"] = df["texto_limpio"].fillna('')
  
  # Preparar los datos
  X = df[final_columns]
  y = df["target"]
  
   
  SEED = 12345
  TEST_SIZE = 0.2
  MAX_WORDS = 10000  # Número máximo de palabras a considerar
  MAX_CLASSES = df['target'].nunique()  # Número de clases en la columna target
  batch_size = 64
  nb_epochs = 20
  
  
  y = to_categorical(df["target"], num_classes=df['target'].nunique())  # Convertir a one-hot encoding

  
  print(f"[{datetime.now()}] - Tokenizacion TFIDF \n")
  tokenizer = Tokenizer(num_words=MAX_WORDS)
  tokenizer.fit_on_texts(df['texto_limpio'].astype(str))  # Asegúrate de que sea tratado como cadena de texto
  X = tokenizer.texts_to_matrix(df['texto_limpio'].astype(str), mode='tfidf')  # Convierte el texto a TF-IDF

  X_df = pd.DataFrame(X, columns=[f'tfidf_{i}' for i in range(X.shape[1])])
  

  # Concatenamos el nuevo DataFrame con el original
  XX = pd.concat([df, X_df], axis=1)
  XX = XX.drop(columns=['texto_limpio'])
 
  

  # Dividir el dataset en train y test
  X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=TEST_SIZE, random_state=SEED)
  total_features = XX.shape[1]
  
  # Crear el modelo
  model = Sequential()
  model.add(Dense(1000, input_shape=(total_features,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(500))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(MAX_CLASSES))
  model.add(Activation('softmax'))

  # Compilar el modelo
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(f"{datetime.now()} - Entrenamos modelo \n")
  # Entrenar el modelo
  model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test), verbose=1)

  # Hacer predicciones
  y_train_predclass = model.predict(X_train, batch_size=batch_size)
  y_test_predclass = model.predict(X_test, batch_size=batch_size)

  # Convertir las etiquetas y las predicciones a formato ordinal
  y_train_labels = np.argmax(y_train, axis=1)
  y_test_labels = np.argmax(y_test, axis=1)
  y_train_pred_labels = np.argmax(y_train_predclass, axis=1)
  y_test_pred_labels = np.argmax(y_test_predclass, axis=1)

  # Calcular la precisión y generar el informe de clasificación
  print("Train accuracy: {}".format(round(accuracy_score(y_train_labels, y_train_pred_labels), 3)))
  print("Test accuracy: {}".format(round(accuracy_score(y_test_labels, y_test_pred_labels), 3)))
  print("\nTest Classification Report\n")
  print(classification_report(y_test_labels, y_test_pred_labels))


def preprocessing(path_df): 
  """
  Eliminamos columnas que no sirven, Imputamos NANs, convertimos Class a numérico y limpiamos el texto.
  """
  df = pd.read_excel(path_df)
  
  df.drop(columns=["DescCuenta","NTesoreria","DescTesoreria","DescEntidad","Beneficiario"], inplace=True)
  
  df["Descripcion"] = df["Descripcion"].fillna("")
  df["ClaseReg"] = df["ClaseReg"].fillna("Indefinido")
  
  
  Class = list(df.Class.unique())
  clases = {val:Class.index(val) for val in Class}
  def get_class(val):
      return clases[val]
    
  df['target'] = df['Class'].apply(get_class)
  df.drop(columns=['Class'], inplace=True)
  df["texto_limpio"] = df["Descripcion"].apply(pre_procesamiento_texto)
  
  return df


def featureEngineering(df):
  """
  Creamos features de texto y features de pesos sobre el texto limpio.
  """
  # features de texto -----------------------------------------------------------------------
  df['Descripcion'] = df['Descripcion'].astype(str)   
  df['description_size'] = df['Descripcion'].str.len()
  df['description_words_count'] = df['Descripcion'].apply(lambda x: len(x.split())) 
  df.drop(columns=['Descripcion'], inplace=True)
  
  df['texto_limpio'] = df['texto_limpio'].astype(str)   
  df['text_size'] = df['texto_limpio'].str.len()
  df['text_words_count'] = df['texto_limpio'].apply(lambda x: len(x.split()))  
  
  categ = ['TipoComp','TipoPres','TipoReg','ClaseReg','TipoCta']
  for col in categ:
      df = pd.concat([df,pd.get_dummies(df[col],prefix=col, prefix_sep='_')],axis=1)
      df.drop(col, axis=1, inplace=True)
  
  # conteo de palabras -----------------------------------------------------------------------
  dictOfWords = {}
  for target in df.target.unique():
    df_target = df[df["target"]==target]
    all_descriptions = ' '.join(df_target['texto_limpio'].dropna())  # Concatenar todas las descripciones

    # Tokenización y eliminación de stopwords
    stop_words = set(stopwords.words('spanish')) 
    word_tokens = word_tokenize(all_descriptions.lower())  # Tokenización y convertir a minúsculas
    filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]  # Filtrar stopwords y no palabras alfa
    freq_of_words = pd.Series(filtered_words).value_counts()
    
    
    dic_words = freq_of_words.to_dict()
    dictOfWords[str(target)] = dic_words
    df[f'pesos_{target}'] = df['texto_limpio'].apply(pesos,dic_words=dic_words)
      
  return df


def pre_procesamiento_texto(text):
  # Quito simbolos
  texto = solo_numeros_y_letras(text)

  # tokenizacion
  texto = separar_texto_de_numeros(texto)

  # Elimino espacios de mas
  texto = eliminar_espacios_adicionales(texto)

  # Elimino stopwords
  texto = remove_stopwords(texto)

  # Lematizacion
  texto = lematizacion(texto)

  # Solo palabras y numeros con minimo de 3 caracteres
  texto = filtrar_palabras_numeros(texto)

  # Eliminar palabras frecuentas: RES-CTA- etc.
  # texto = eliminar_palabras(texto)

  return texto


def solo_numeros_y_letras(text):
  # Reemplazar todo lo que no sea letra o número por un espacio
  text_limpio = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ]', ' ', str(text))
  return text_limpio

def separar_texto_de_numeros(texto):
    # Expresión regular para insertar un espacio entre letras y números
    texto = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', texto)
    texto = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', texto)
    return texto

def eliminar_espacios_adicionales(texto):
    # Reemplazar múltiples espacios por un solo espacio
    texto_limpio = ' '.join(texto.split())
    return texto_limpio
  

def remove_stopwords(text):
    """
    Elimino stopwords en español.
    """
    return ' '.join([word for word in text.split() if word.lower() not in STOPWORDS])


def lematizacion(text):
  """
  no stop words + lematizacion
  """
  clean_text = []
  
  for token in nlp(text):
    if (
        not token.is_stop             # Excluir stop words
        and (token.is_alpha or token.is_digit)  # Incluir solo letras o números
        and not token.is_punct        # Excluir signos de puntuación
        and not token.like_url        # Excluir URLs
    ):
        clean_text.append(token.lemma_.upper())  

  return " ".join(clean_text)

def filtrar_palabras_numeros(texto):
    # Expresión regular para encontrar palabras o números con al menos 3 caracteres
    palabras_filtradas = re.findall(r'\b\w{3,}\b', texto)
    return " ".join(palabras_filtradas)
  

def eliminar_palabras(texto):
  texto = texto.upper()
  texto = re.findall(r"(?!CTA)(?!RES)(?!PAGO)(?!AGO)(?!PAG)[A-Z0-9]{3,}", texto)
  texto = list(dict.fromkeys(texto))
  texto = " ".join(texto).strip()
  return texto
  
def pesos(texto, dic_words):
  texto = texto.lower()
  palabras = texto.split(' ')
  score = 0
  for palabra in palabras:
      if palabra in dic_words.keys():
          score += dic_words[palabra]
  return score


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


