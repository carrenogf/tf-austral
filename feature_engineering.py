import os
import pandas as pd
import numpy as np
import json
import traceback
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler


TOKEN_PATTERN = re.compile(r"\b\w{2,}\b", flags=re.UNICODE)


def fe(dataset_dir="./datasets"):
  try:
    dataset_dir = os.path.abspath(dataset_dir)
    print("="*60)
    print("Iniciando Feature Engineering sobre train/val/test...")
    print("="*60)
    
    # leemos
    print("\n[1/8] Leyendo datos preprocesados...")
    paths_in = {
      "train": os.path.join(dataset_dir, "df_preprocessed_train.csv"),
      "val": os.path.join(dataset_dir, "df_preprocessed_val.csv"),
      "test": os.path.join(dataset_dir, "df_preprocessed_test.csv"),
    }
    dfs = {split: pd.read_csv(path, sep=';') for split, path in paths_in.items()}
    for split, df in dfs.items():
      print(f"   ✓ {split}: {len(df)} registros, {len(df.columns)} columnas")
    
    # Creamos variables de texto
    print("\n[2/8] Creando variables de texto...")
    for split in dfs:
      dfs[split] = crear_variablesTexto(dfs[split])
    print("   ✓ Variables creadas para train/val/test")
    
    # eliminamos variables que no sirven
    print("\n[3/8] Eliminando variables innecesarias...")
    for split in dfs:
      dfs[split].drop(columns=['descripcion'], inplace=True)
      print(f"   ✓ {split}: columna 'descripcion' eliminada. Columnas restantes: {len(dfs[split].columns)}")

    print("\n[4/8] OneHotEncoding ya fue aplicado en el preprocesamiento")
        
    # conteo de palabras
    print("\n[5/8] Calculando diccionario de palabras (solo train)...")
    dict_words, targets = build_word_dictionary(dfs["train"], min_word_freq=3, top_k=5000, alpha=1.0)
    dict_path = os.path.join(dataset_dir, "dict_words_train.json")
    with open(dict_path, 'w') as file:
      json.dump(dict_words, file, indent=4)
    print(f"   ✓ Diccionario guardado en {dict_path}")

    # conteo de pares de palabras (bigramas)
    print("\n[6/8] Calculando diccionario de pares de palabras (solo train)...")
    dict_bigrams, bigram_targets = build_bigram_dictionary(
      dfs["train"],
      min_bigram_freq=3,
      top_k=5000,
      alpha=1.0
    )
    dict_bigrams_path = os.path.join(dataset_dir, "dict_bigrams_train.json")
    with open(dict_bigrams_path, 'w') as file:
      json.dump(dict_bigrams, file, indent=4)
    print(f"   ✓ Diccionario de bigramas guardado en {dict_bigrams_path}")

    print("\n[7/8] Asignando pesos al texto...")
    for split in dfs:
      dfs[split] = asignar_pesos_al_texto(dfs[split], dict_words, targets)
      dfs[split] = asignar_pesos_bigrams_al_texto(dfs[split], dict_bigrams, bigram_targets)
      print(f"   ✓ Pesos asignados para {split}")
    
    # Estandarizacion de pesos
    print("\n[8/8] Estandarizando pesos con scaler del train...")
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
  df['descripcion'] = df['descripcion'].astype(str)
  df['texto_limpio'] = df['texto_limpio'].astype(str)
  
  df['descripcion_size'] = df['descripcion'].str.len()
  df['descripcion_words_count'] = df['descripcion'].apply(lambda x: len(x.split()))
  df['descripcion_unique_words_count'] = df['descripcion'].apply(lambda x: len(set(x.split())))
  df['descripcion_is_empty'] = (df['descripcion_words_count'] == 0).astype(int)

  # Señales de formato en descripción original
  df['descripcion_digit_count'] = df['descripcion'].str.count(r'\d')
  df['descripcion_digit_ratio'] = _safe_divide(df['descripcion_digit_count'], df['descripcion_size'])
  df['descripcion_upper_ratio'] = _safe_divide(
    df['descripcion'].str.count(r'[A-ZÁÉÍÓÚÑ]'),
    df['descripcion_size']
  )
  df['descripcion_special_ratio'] = _safe_divide(
    df['descripcion'].str.count(r'[^\w\s]'),
    df['descripcion_size']
  )
  df['descripcion_has_year'] = df['descripcion'].str.contains(
    r'\b(?:19|20)\d{2}\b', regex=True
  ).astype(int)
  df['descripcion_has_currency'] = df['descripcion'].str.contains(
    r'(?:\$|usd|u\$s|ars|dolar|dólar)', case=False, regex=True
  ).astype(int)

  df['text_size'] = df['texto_limpio'].str.len()
  df['text_words_count'] = df['texto_limpio'].apply(lambda x: len(x.split()))
  df['text_unique_words_count'] = df['texto_limpio'].apply(lambda x: len(set(x.split())))

  token_lists = df['texto_limpio'].str.split()
  df['text_long_tokens_count'] = token_lists.apply(lambda toks: sum(len(t) >= 8 for t in toks))
  df['text_short_tokens_count'] = token_lists.apply(lambda toks: sum(len(t) <= 3 for t in toks))
  df['text_token_len_mean'] = token_lists.apply(
    lambda toks: float(np.mean([len(t) for t in toks])) if toks else 0.0
  )
  df['text_token_len_std'] = token_lists.apply(
    lambda toks: float(np.std([len(t) for t in toks])) if toks else 0.0
  )

  # Métricas robustas para capturar densidad y diversidad del texto
  df['text_avg_word_len'] = _safe_divide(df['text_size'], df['text_words_count'])
  df['text_lexical_diversity'] = _safe_divide(df['text_unique_words_count'], df['text_words_count'])
  df['text_digit_ratio'] = df['texto_limpio'].apply(_digit_token_ratio)
  df['text_has_digits'] = (df['text_digit_ratio'] > 0).astype(int)
  df['text_long_token_ratio'] = _safe_divide(df['text_long_tokens_count'], df['text_words_count'])
  df['text_short_token_ratio'] = _safe_divide(df['text_short_tokens_count'], df['text_words_count'])
  df['text_repeated_token_ratio'] = (1.0 - df['text_lexical_diversity']).clip(0.0, 1.0)
  
  print(
    "   - Variables creadas: descripcion_size, descripcion_words_count, descripcion_unique_words_count, "
    "descripcion_is_empty, descripcion_digit_count, descripcion_digit_ratio, descripcion_upper_ratio, "
    "descripcion_special_ratio, descripcion_has_year, descripcion_has_currency, text_size, text_words_count, "
    "text_unique_words_count, texto_limpio_is_empty, text_long_tokens_count, text_short_tokens_count, "
    "text_token_len_mean, text_token_len_std, text_avg_word_len, text_lexical_diversity, text_digit_ratio, "
    "text_has_digits, text_long_token_ratio, text_short_token_ratio, text_repeated_token_ratio"
  )
  return df


def _safe_divide(numerator, denominator):
  """
  División segura entre series (retorna 0 cuando el denominador es 0).
  """
  denom = denominator.replace(0, np.nan)
  return (numerator / denom).fillna(0.0).astype(float)



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


def _tokenize(texto):
  """
  Tokenizador liviano y estable (sin depender de recursos externos de NLTK).
  """
  if pd.isna(texto):
    return []
  return TOKEN_PATTERN.findall(str(texto).lower())


def _digit_token_ratio(texto):
  """
  Ratio de tokens numéricos sobre total de tokens.
  """
  tokens = str(texto).split()
  if not tokens:
    return 0.0
  digit_tokens = sum(tok.isdigit() for tok in tokens)
  return float(digit_tokens) / float(len(tokens))


def _to_bigrams(tokens):
  """
  Convierte una lista de tokens en bigramas consecutivos "token1 token2".
  """
  if len(tokens) < 2:
    return []
  return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def pesos(texto, dic_words):
  """
  Puntaje de texto para una clase: suma de pesos de tokens presentes.
  """
  score = 0.0
  for palabra in _tokenize(texto):
    score += dic_words.get(palabra, 0.0)
  return float(score)


def pesos_bigrams(texto, dic_bigrams):
  """
  Puntaje de texto para una clase: suma de pesos de bigramas presentes.
  """
  score = 0.0
  for bigrama in _to_bigrams(_tokenize(texto)):
    score += dic_bigrams.get(bigrama, 0.0)
  return float(score)



def build_word_dictionary(df_train, min_word_freq=3, top_k=5000, alpha=1.0):
  """
  Construye diccionario de palabras por clase usando solo train,
  con peso log-odds para reducir sesgo por clases más frecuentes.
  """
  dictOfWords = {}
  targets = sorted(df_train.target.unique())

  token_lists = df_train['texto_limpio'].fillna('').apply(_tokenize)
  global_counter = Counter()
  class_counters = {}

  for idx, target in enumerate(targets, 1):
    print(f"   - Procesando clase {target} ({idx}/{len(targets)})...")
    class_tokens = [tok for toks in token_lists[df_train['target'] == target] for tok in toks]
    class_counter = Counter(class_tokens)
    class_counters[target] = class_counter
    global_counter.update(class_counter)

  vocab = [w for w, c in global_counter.items() if c >= min_word_freq]
  vocab_size = max(len(vocab), 1)
  all_words_set = set(vocab)
  total_all = sum(global_counter[w] for w in all_words_set)

  for target in targets:
    class_counter = class_counters[target]
    total_class = sum(class_counter[w] for w in all_words_set)
    total_other = max(total_all - total_class, 0)

    # Priorizamos palabras más representativas de la clase
    words_by_freq = [w for w, _ in class_counter.most_common(top_k) if w in all_words_set]
    word_scores = {}
    for word in words_by_freq:
      c_t = class_counter.get(word, 0)
      c_not_t = global_counter.get(word, 0) - c_t

      # log P(w|clase) - log P(w|resto), con smoothing
      p_t = (c_t + alpha) / (total_class + alpha * vocab_size)
      p_not_t = (c_not_t + alpha) / (total_other + alpha * vocab_size)
      word_scores[word] = float(np.log(p_t) - np.log(p_not_t))

    dictOfWords[str(target)] = word_scores

  return dictOfWords, targets


def build_bigram_dictionary(df_train, min_bigram_freq=3, top_k=5000, alpha=1.0):
  """
  Construye diccionario de bigramas por clase usando solo train,
  con peso log-odds para reducir sesgo por clases más frecuentes.
  """
  dictOfBigrams = {}
  targets = sorted(df_train.target.unique())

  token_lists = df_train['texto_limpio'].fillna('').apply(_tokenize)
  bigram_lists = token_lists.apply(_to_bigrams)

  global_counter = Counter()
  class_counters = {}

  for idx, target in enumerate(targets, 1):
    print(f"   - Procesando clase {target} ({idx}/{len(targets)})...")
    class_bigrams = [bg for bgs in bigram_lists[df_train['target'] == target] for bg in bgs]
    class_counter = Counter(class_bigrams)
    class_counters[target] = class_counter
    global_counter.update(class_counter)

  vocab = [bg for bg, c in global_counter.items() if c >= min_bigram_freq]
  vocab_size = max(len(vocab), 1)
  all_bigrams_set = set(vocab)
  total_all = sum(global_counter[bg] for bg in all_bigrams_set)

  for target in targets:
    class_counter = class_counters[target]
    total_class = sum(class_counter[bg] for bg in all_bigrams_set)
    total_other = max(total_all - total_class, 0)

    # Priorizamos bigramas más representativos de la clase
    bigrams_by_freq = [bg for bg, _ in class_counter.most_common(top_k) if bg in all_bigrams_set]
    bigram_scores = {}
    for bigram in bigrams_by_freq:
      c_t = class_counter.get(bigram, 0)
      c_not_t = global_counter.get(bigram, 0) - c_t

      # log P(bg|clase) - log P(bg|resto), con smoothing
      p_t = (c_t + alpha) / (total_class + alpha * vocab_size)
      p_not_t = (c_not_t + alpha) / (total_other + alpha * vocab_size)
      bigram_scores[bigram] = float(np.log(p_t) - np.log(p_not_t))

    dictOfBigrams[str(target)] = bigram_scores

  return dictOfBigrams, targets


def asignar_pesos_al_texto(df, dict_words, targets):    
  """
  Usa un diccionario pre-calculado (solo train) para generar pesos en cualquier split.
  """
  print("   (Esto puede tomar varios minutos)")
  pesos_cols = []
  for target in targets:
    word_dict = dict_words.get(str(target), {})
    col = f'pesos_{target}'
    df[col] = df['texto_limpio'].apply(pesos, dic_words=word_dict)
    pesos_cols.append(col)

  if pesos_cols:
    df['pesos_max'] = df[pesos_cols].max(axis=1)
    if len(pesos_cols) >= 2:
      top2 = np.sort(df[pesos_cols].to_numpy(), axis=1)[:, -2:]
      df['pesos_margin'] = top2[:, 1] - top2[:, 0]
    else:
      df['pesos_margin'] = df['pesos_max']

  return df


def asignar_pesos_bigrams_al_texto(df, dict_bigrams, targets):
  """
  Usa un diccionario pre-calculado de bigramas (solo train) para generar pesos en cualquier split.
  """
  print("   (Calculando pesos de bigramas)")
  pesos_cols = []
  for target in targets:
    bigram_dict = dict_bigrams.get(str(target), {})
    col = f'pesos_bigram_{target}'
    df[col] = df['texto_limpio'].apply(pesos_bigrams, dic_bigrams=bigram_dict)
    pesos_cols.append(col)

  if pesos_cols:
    df['pesos_bigram_max'] = df[pesos_cols].max(axis=1)
    if len(pesos_cols) >= 2:
      top2 = np.sort(df[pesos_cols].to_numpy(), axis=1)[:, -2:]
      df['pesos_bigram_margin'] = top2[:, 1] - top2[:, 0]
    else:
      df['pesos_bigram_margin'] = df['pesos_bigram_max']

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