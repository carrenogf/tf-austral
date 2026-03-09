# Librerias basicas 
import os
import re
import spacy
from nltk.corpus import stopwords
import pandas as pd
import traceback
import json

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


import warnings
warnings.filterwarnings("ignore")


def preprocess(dataset_dir="./datasets", train_file="2022-2023.csv", val_file="2024.csv", test_file="2025.csv"):
  """
  Preprocesa train/val/test usando el mismo mapeo de clases, columnas de OHE y pipeline de texto.
  Se generan df_preprocessed_train/val/test en la carpeta datasets.
  """
  try:
    dataset_dir = os.path.abspath(dataset_dir)
    paths_in = {
      "train": os.path.join(dataset_dir, train_file),
      "val": os.path.join(dataset_dir, val_file),
      "test": os.path.join(dataset_dir, test_file),
    }

    print("="*60)
    print("Iniciando preprocesamiento de train/val/test...")
    print("="*60)

    # leer datos
    dfs = {}
    for split, path in paths_in.items():
      print(f"\n[1/6] Leyendo {split} desde: {path}")
      dfs[split] = pd.read_csv(path)
      print(f"   ✓ {split}: {len(dfs[split])} registros, {len(dfs[split].columns)} columnas")

    # # Eliminar columnas que no sirven (se ajusta sobre train y se replica)
    # drop_cols = ["DescCuenta","NTesoreria","DescTesoreria","DescEntidad","Beneficiario","Cuit"]
    # print("\n[2/6] Eliminando columnas innecesarias...")
    # for split in dfs:
    #   dfs[split] = eliminar_columnas(dfs[split], drop_cols)
    #   print(f"   ✓ {split}: {len(dfs[split].columns)} columnas")

    # seleccionar columnas necesarias
    required_cols = ["tipo_comp",	"nro_cuenta","nro_entidad",	"tipo_pres",	"tipo_reg",	"clase_reg",	"cod",
                      "fuente_fin",	"descripcion",	"tipo_cta",	"cod_bco",	"Class"]
    
    print("\n[2/6] Seleccionando columnas necesarias...")

    for split in dfs:
      dfs[split] = dfs[split][required_cols]
      print(f"   ✓ {split}: {len(dfs[split].columns)} columnas")


    # Imputar NA's
    print("\n[3/6] Imputando valores NA...")
    for split in dfs:
      dfs[split] = imputarNA(dfs[split])
    print("   ✓ Valores NA imputados en todos los splits")

    # "Class" de String a Entero usando solo train
    print("\n[4/6] Convirtiendo 'Class' a valores numéricos con el mapeo del train...")
    dfs["train"], class_mapping = ClassToInt(dfs["train"], class_mapping=None)
    dfs["val"], _ = ClassToInt(dfs["val"], class_mapping=class_mapping)
    dfs["test"], _ = ClassToInt(dfs["test"], class_mapping=class_mapping)
    print(f"   ✓ Mapeo de clases: {class_mapping}")

    # Preprocesamiento del texto
    print("\n[5/6] Aplicando preprocesamiento de texto...")
    print("   (Esto puede tomar varios minutos)")
    for split in dfs:
      dfs[split]["texto_limpio"] = dfs[split]["descripcion"].apply(pre_procesamiento_texto)
    print("   ✓ Texto procesado para train/val/test")

    # OneHotEncoding de variables categóricas con columnas definidas por train
    print("\n[6/6] Aplicando OneHotEncoding a variables categóricas con base en train...")
    dfs["train"], dfs["val"], dfs["test"] = aplicar_ohe_splits(
      dfs["train"], dfs["val"], dfs["test"],
      columnas=['tipo_comp','tipo_reg','clase_reg','tipo_cta']
    )
    print("   ✓ OneHotEncoding completado")

    # Guardamos
    os.makedirs(dataset_dir, exist_ok=True)
    paths_out = {
      "train": os.path.join(dataset_dir, "df_preprocessed_train.csv"),
      "val": os.path.join(dataset_dir, "df_preprocessed_val.csv"),
      "test": os.path.join(dataset_dir, "df_preprocessed_test.csv"),
    }
    print("\n[7/6] Guardando datos preprocesados...")
    for split, df in dfs.items():
      df.to_csv(paths_out[split], index=False, sep=';')
      print(f"   ✓ {split}: {paths_out[split]}")

    # Guardamos el mapeo de clases para uso posterior
    class_map_path = os.path.join(dataset_dir, "class_mapping.json")
    with open(class_map_path, "w") as f:
      json.dump(class_mapping, f, indent=2)
    print(f"   ✓ Mapeo de clases guardado en {class_map_path}")

    print("\n" + "="*60)
    print("¡Preprocesamiento completado exitosamente!")
    print("="*60)

  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")



def eliminar_columnas(df, columnas):  
  """
  Eliminamos columnas que no sirven.
  """
  print(f"   - Eliminando: {', '.join(columnas)}")
  df.drop(columns=[c for c in columnas if c in df.columns], inplace=True)
  return df


def imputarNA(df):
  """
  """
  na_descripcion = df["descripcion"].isna().sum()
  na_clasereg = df["clase_reg"].isna().sum()
  print(f"   - NAs en 'descripcion': {na_descripcion}")
  print(f"   - NAs en 'clase_reg': {na_clasereg}")
  df["descripcion"] = df["descripcion"].fillna("")
  df["clase_reg"] = df["clase_reg"].fillna("Indefinido")
  return df


def ClassToInt(df, class_mapping=None):
  """
  Mapea la columna Class a enteros. Si se pasa class_mapping, se reutiliza, de lo contrario se crea con el train.
  """
  if class_mapping is None:
    class_values = list(df.Class.unique())
    class_mapping = {val: idx for idx, val in enumerate(class_values)}
    print(f"   - Clases encontradas (train): {class_values}")
  else:
    print(f"   - Usando mapeo de clases existente: {class_mapping}")

  df['target'] = df['Class'].map(class_mapping)
  unknown = df['target'].isna().sum()
  if unknown > 0:
    print(f"   ! {unknown} registros con clases desconocidas fueron asignados a -1")
    df['target'] = df['target'].fillna(-1)
  df['target'] = df['target'].astype(int)
  df.drop(columns=['Class'], axis=1, inplace=True)
  return df, class_mapping

def pre_procesamiento_texto(text):
  """

  """
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


def aplicar_ohe_splits(df_train, df_val, df_test, columnas):
  """
  Ajusta OHE en train y replica la estructura en val y test.
  """
  print(f"   - Columnas para OneHotEncoding: {columnas}")

  train_ohe = pd.get_dummies(df_train, columns=columnas, prefix=columnas, prefix_sep='_')
  val_ohe = pd.get_dummies(df_val, columns=columnas, prefix=columnas, prefix_sep='_')
  test_ohe = pd.get_dummies(df_test, columns=columnas, prefix=columnas, prefix_sep='_')

  base_cols = train_ohe.columns

  def _align(df, base_cols):
    missing = [c for c in base_cols if c not in df.columns]
    for col in missing:
      df[col] = 0
    extra = [c for c in df.columns if c not in base_cols]
    if extra:
      df.drop(columns=extra, inplace=True)
    return df[base_cols]

  val_ohe = _align(val_ohe, base_cols)
  test_ohe = _align(test_ohe, base_cols)

  print(f"   - OneHotEncoding completado. Columnas finales: {len(base_cols)}")
  return train_ohe, val_ohe, test_ohe