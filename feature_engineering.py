import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import traceback
from sklearn.preprocessing import StandardScaler


def fe(dataset_dir="./datasets"):
  try:
    dataset_dir = os.path.abspath(dataset_dir)
    print("="*60)
    print("Iniciando Feature Engineering sobre train/val/test...")
    print("="*60)
    
    # leemos
    print("\n[1/7] Leyendo datos preprocesados...")
    paths_in = {
      "train": os.path.join(dataset_dir, "df_preprocessed_train.csv"),
      "val": os.path.join(dataset_dir, "df_preprocessed_val.csv"),
      "test": os.path.join(dataset_dir, "df_preprocessed_test.csv"),
    }
    dfs = {split: pd.read_csv(path, sep=';') for split, path in paths_in.items()}
    for split, df in dfs.items():
      print(f"   ✓ {split}: {len(df)} registros, {len(df.columns)} columnas")
    
    # Creamos variables de texto
    print("\n[2/7] Creando variables de texto...")
    for split in dfs:
      dfs[split] = crear_variablesTexto(dfs[split])
    print("   ✓ Variables creadas para train/val/test")
    
    # eliminamos variables que no sirven
    print("\n[3/7] Eliminando variables innecesarias...")
    for split in dfs:
      dfs[split].drop(columns=['Descripcion'], inplace=True)
      print(f"   ✓ {split}: columna 'Descripcion' eliminada. Columnas restantes: {len(dfs[split].columns)}")

    print("\n[4/7] OneHotEncoding ya fue aplicado en el preprocesamiento")
        
    # conteo de palabras
    print("\n[5/7] Calculando diccionario de palabras (solo train)...")
    dict_words, targets = build_word_dictionary(dfs["train"])
    dict_path = os.path.join(dataset_dir, "dict_words_train.json")
    with open(dict_path, 'w') as file:
      json.dump(dict_words, file, indent=4)
    print(f"   ✓ Diccionario guardado en {dict_path}")

    print("\n[6/7] Asignando pesos al texto...")
    for split in dfs:
      dfs[split] = asignar_pesos_al_texto(dfs[split], dict_words, targets)
      print(f"   ✓ Pesos asignados para {split}")
    
    # Estandarizacion de pesos
    print("\n[7/7] Estandarizando pesos con scaler del train...")
    dfs["train"], scaler = estandarizar_pesos(dfs["train"], scaler=None)
    dfs["val"], _ = estandarizar_pesos(dfs["val"], scaler=scaler)
    dfs["test"], _ = estandarizar_pesos(dfs["test"], scaler=scaler)
    
    # Alinear columnas con base en train por si falta algun peso
    train_cols = dfs["train"].columns
    dfs["val"] = _align_to_base(dfs["val"], train_cols)
    dfs["test"] = _align_to_base(dfs["test"], train_cols)

    # Guardamos
    paths_out = {
      "train": os.path.join(dataset_dir, "df_final_train.csv"),
      "val": os.path.join(dataset_dir, "df_final_val.csv"),
      "test": os.path.join(dataset_dir, "df_final_test.csv"),
    }
    print("\nGuardando datos finales...")
    for split, df in dfs.items():
      df.to_csv(paths_out[split], index=False, sep=';')
      print(f"   ✓ {split}: {paths_out[split]}")
    
    print("\n" + "="*60)
    print("¡Feature Engineering completado exitosamente!")
    print("="*60)
    return dfs

  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")
    return None
  

def crear_variablesTexto(df):
  """
  """
  df['Descripcion'] = df['Descripcion'].astype(str)
  df['texto_limpio'] = df['texto_limpio'].astype(str)
  
  df['descripcion_size'] = df['Descripcion'].str.len()
  df['descripcion_words_count'] = df['Descripcion'].apply(lambda x: len(x.split()))  

  df['text_size'] = df['texto_limpio'].str.len()
  df['text_words_count'] = df['texto_limpio'].apply(lambda x: len(x.split()))
  
  print(f"   - Variables creadas: descripcion_size, descripcion_words_count, text_size, text_words_count")
  return df



# Función de OneHotEncoding movida al preprocesamiento
# No se usa aquí porque ya se aplica en preprocessing.py
# def aplicar_ohe(df):
#   """
#   """
#   categ = ['TipoComp','TipoReg','ClaseReg','TipoCta']
#   print(f"   - Columnas para One-Hot Encoding: {categ}")
#   for col in categ:
#       df = pd.concat([df,pd.get_dummies(df[col],prefix=col, prefix_sep='_')],axis=1)
#       df.drop(col, axis=1, inplace=True)
#   print(f"   - One-Hot Encoding completado")
#   return df


def pesos(texto, dic_words):
  """
  """
  texto = texto.lower()
  palabras = texto.split(' ')
  score = 0
  for palabra in palabras:
      if palabra in dic_words.keys():
          score += dic_words[palabra]
  return score



def build_word_dictionary(df_train):
  """
  Construye diccionario de palabras por clase usando solo train.
  """
  dictOfWords = {}
  targets = sorted(df_train.target.unique())
  for idx, target in enumerate(targets, 1):
      print(f"   - Procesando clase {target} ({idx}/{len(targets)})...")
      df_target = df_train[df_train["target"]==target]
      all_descriptions = ' '.join(df_target['texto_limpio'].dropna())

      stop_words = set(stopwords.words('spanish'))
      word_tokens = word_tokenize(all_descriptions.lower())
      filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]
      freq_of_words = pd.Series(filtered_words).value_counts()
      dictOfWords[str(target)] = freq_of_words.to_dict()
  return dictOfWords, targets


def asignar_pesos_al_texto(df, dict_words, targets):    
  """
  Usa un diccionario pre-calculado (solo train) para generar pesos en cualquier split.
  """
  print("   (Esto puede tomar varios minutos)")
  for target in targets:
      word_dict = dict_words.get(str(target), {})
      df[f'pesos_{target}'] = df['texto_limpio'].apply(pesos, dic_words=word_dict)
  return df


def estandarizar_pesos(df, scaler=None):
  """
  Estandariza columnas de pesos usando scaler entrenado en train.
  """
  pesos_cols = [col for col in df.columns if col.startswith('pesos_')]
  print(f"   - Columnas a estandarizar: {pesos_cols}")
  if not pesos_cols:
    return df, scaler
  if scaler is None:
    scaler = StandardScaler()
    df[pesos_cols] = scaler.fit_transform(df[pesos_cols])
  else:
    df[pesos_cols] = scaler.transform(df[pesos_cols])
  print(f"   ✓ Pesos estandarizados")
  return df, scaler


def _align_to_base(df, base_cols):
  """
  Alinea columnas de val/test a las del train (agrega faltantes y elimina extras).
  """
  missing = [c for c in base_cols if c not in df.columns]
  for col in missing:
    df[col] = 0
  extra = [c for c in df.columns if c not in base_cols]
  if extra:
    df.drop(columns=extra, inplace=True)
  return df[base_cols]