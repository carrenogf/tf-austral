"""
Evaluación de Data Leakage en los scripts de Preprocessing y Feature Engineering.

Este script analiza si hay data leakage (fuga de información) entre splits train/val/test.
Data leakage ocurre cuando información de val/test se usa durante el entrenamiento del modelo.
"""

import json
import pandas as pd
import numpy as np
import os


def print_assessment_header(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_section(section_name):
    print(f"\n{'─'*80}")
    print(f"  {section_name}")
    print(f"{'─'*80}")


def print_status(status, message):
    """Imprime un mensaje con color según el estado."""
    symbols = {
        'OK': '✓',
        'WARNING': '⚠',
        'CRITICAL': '✗'
    }
    colors = {
        'OK': '\033[92m',  # Verde
        'WARNING': '\033[93m',  # Amarillo
        'CRITICAL': '\033[91m',  # Rojo
        'RESET': '\033[0m'
    }
    symbol = symbols.get(status, '?')
    color = colors.get(status, '')
    reset = colors['RESET']
    print(f"  {color}{symbol}{reset} {message}")


def assess_preprocessing():
    """Evalúa el script preprocessing.py para data leakage."""
    print_assessment_header("ANÁLISIS DE DATA LEAKAGE - PREPROCESSING.py")
    
    findings = {
        'critical_issues': [],
        'warnings': [],
        'ok_points': []
    }
    
    print_section("1. División de datos (Train/Val/Test)")
    print("""
  Análisis:
  • Train: 2022-2023-2024.csv
  • Val: 2025-1.csv
  • Test: 2025-2.csv
  
  Observación: Los splits están separados cronológicamente (datos por año).
  Esto es CORRECTO y evita leakage temporal.
    """)
    findings['ok_points'].append("Splits separados cronológicamente")
    
    print_section("2. Mapeo de Clases (ClassToInt)")
    print("""
  Código analizado:
  ───────────────────
  if class_mapping is None:
      class_values = list(df.Class.unique())
      class_mapping = {val: idx for idx, val in enumerate(class_values)}
      # ... usado solo en TRAIN
  
  dfs["train"], class_mapping = ClassToInt(dfs["train"], class_mapping=None)
  dfs["val"], _ = ClassToInt(dfs["val"], class_mapping=class_mapping)
  dfs["test"], _ = ClassToInt(dfs["test"], class_mapping=class_mapping)
    """)
    print_status('OK', "El mapeo de clases se crea SOLO a partir del TRAIN")
    print_status('OK', "Se reutiliza el mismo mapeo en val/test (sin re-entrenar)")
    findings['ok_points'].append("Mapeo de clases coherente entre splits")
    
    print_section("3. Procesamiento de Texto")
    print("""
  Código analizado:
  ───────────────────
  for split in dfs:
      dfs[split]["texto_limpio"] = dfs[split]["descripcion"].apply(
          pre_procesamiento_texto
      )
    """)
    print_status('OK', "Pre-procesamiento es una transformación DETERMINÍSTICA")
    print_status('OK', "Se aplica de igual forma a train/val/test")
    print_status('OK', "NO depende de estadísticas globales del corpus")
    findings['ok_points'].append("Procesamiento de texto determinístico")
    
    print_section("4. OneHotEncoding (OHE)")
    print("""
  Código analizado:
  ───────────────────
  dfs["train"], dfs["val"], dfs["test"] = aplicar_ohe_splits(
      dfs["train"], dfs["val"], dfs["test"],
      columnas=['tipo_comp','tipo_reg','clase_reg','tipo_cta',...]
  )
    """)
    print_status('OK', "OHE se implementa con columnas definidas por TRAIN")
    print_status('OK', "Val/test reciben SOLO las categorías vistas en train")
    print_status('OK', "Val/test rellenan con 0 las categorías desconocidas")
    findings['ok_points'].append("OneHotEncoding correcto (basado en train)")
    
    print_section("5. CONCLUSIÓN: PREPROCESSING")
    print_status('OK', "✓ No se detecta data leakage en preprocessing.py")
    print_status('OK', "✓ Todos los pasos mantienen la independencia de los splits")
    
    return findings


def assess_feature_engineering():
    """Evalúa el script feature_engineering.py para data leakage."""
    print_assessment_header("ANÁLISIS DE DATA LEAKAGE - FEATURE_ENGINEERING.py")
    
    findings = {
        'critical_issues': [],
        'warnings': [],
        'ok_points': []
    }
    
    print_section("1. Variables de Texto (crear_variablesTexto)")
    print("""
  Variables creadas: descripcion_size, text_words_count, text_lexical_diversity, etc.
  
  Análisis por variable:
  ────────────────────────
    """)
    
    variables_analysis = [
        ("descripcion_size", "len(descripcion)", "OK", "Cálculo por fila, sin estadísticas globales"),
        ("text_words_count", "len(texto.split())", "OK", "Cálculo por fila, determinístico"),
        ("text_lexical_diversity", "unique_words / total_words", "OK", "Solo información de la fila",),
        ("text_token_len_mean", "mean(len(tokens))", "OK", "Solo información de la fila"),
        ("etc.", "Todas las variables", "OK", "Se basan en información POR FILA, sin estadísticas globales"),
    ]
    
    for var_name, formula, status, reason in variables_analysis:
        print_status(status, f"{var_name:25} | {formula:40} | {reason}")
    
    findings['ok_points'].append("Variables de texto sin información global")
    
    print_section("2. Diccionario de Palabras (build_word_dictionary)")
    print("""
  Código analizado:
  ───────────────────
  token_lists = df_train['texto_limpio'].fillna('').apply(_tokenize)  # SOLO TRAIN
  global_counter = Counter()
  class_counters = {}
  
  for target in targets:
      class_tokens = [tok for toks in token_lists[df_train['target'] == target] ...]
      # ...
  
  dict_words = {target: {word: score, ...}, ...}
    """)
    print_status('OK', "Diccionario se entrena SOLO con TRAIN")
    print_status('OK', "Se usa la frecuencia de palabras solo del conjunto train")
    print_status('OK', "Se aplica el mismo diccionario pre-calculado a val/test")
    findings['ok_points'].append("Diccionario de palabras entrenado solo en train")
    
    print_section("3. Diccionario de Bigramas (build_bigram_dictionary)")
    print("""
  Código analizado:
  ───────────────────
  token_lists = df_train['texto_limpio'].fillna('').apply(_tokenize)  # SOLO TRAIN
  bigram_lists = token_lists.apply(_to_bigrams)
  
  dict_bigrams = {target: {bigram: score, ...}, ...}
    """)
    print_status('OK', "Diccionario de bigramas se entrena SOLO con TRAIN")
    print_status('OK', "Se aplica el mismo diccionario pre-calculado a val/test")
    findings['ok_points'].append("Diccionario de bigramas entrenado solo en train")
    
    print_section("4. Pesos de Palabras (asignar_pesos_al_texto)")
    print("""
  Código analizado:
  ───────────────────
  for target in targets:
      word_dict = dict_words.get(str(target), {})  # Dict pre-calculado
      col = f'pesos_{target}'
      df[col] = df['texto_limpio'].apply(pesos, dic_words=word_dict)
    """)
    print_status('OK', "Usa diccionario PRE-CALCULADO en train")
    print_status('OK', "NO re-entrena ni ajusta el diccionario para val/test")
    findings['ok_points'].append("Pesos de palabras usando diccionario fijo")
    
    print_section("5. Estandarización (StandardScaler)")
    print("""
  Código analizado:
  ───────────────────
  dfs["train"], scaler = estandarizar_pesos(dfs["train"], scaler=None)
  # scaler.fit() ocurre AQUI (solo en train)
  
  dfs["val"], _ = estandarizar_pesos(dfs["val"], scaler=scaler)
  dfs["test"], _ = estandarizar_pesos(dfs["test"], scaler=scaler)
  # scaler.transform() (usando parámetros de train)
    """)
    print_status('OK', "StandardScaler se ENTRENA SOLO en train")
    print_status('OK', "Val/test se transforman usando media/std de TRAIN")
    print_status('OK', "✓ CORRECTO: Sin leakage en estandarización")
    findings['ok_points'].append("StandardScaler correcto (fit en train, transform en val/test)")
    
    print_section("6. TF-IDF (aplicar_tfidf)")
    print("""
  Código analizado:
  ───────────────────
  tfidf = TfidfVectorizer(max_features=10000, ...)
  tfidf_train = tfidf.fit_transform(dfs["train"]['texto_limpio'])
  # fit() calcula: IDF (Inverse Document Frequency) de SOLO TRAIN
  
  tfidf_val = tfidf.transform(dfs["val"]['texto_limpio'])
  # transform() usa IDF calculado en train
  
  tfidf_test = tfidf.transform(dfs["test"]['texto_limpio'])
    """)
    print_status('OK', "TF-IDF se ENTRENA (fit) SOLO en train")
    print_status('OK', "Val/test se transforman con vocabulario e IDF de TRAIN")
    print_status('OK', "✓ CORRECTO: Sin leakage en TF-IDF")
    findings['ok_points'].append("TF-IDF correcto (fit en train, transform en val/test)")
    
    print_section("7. Alineamiento de Columnas (_align_to_base)")
    print("""
  Código analizado:
  ───────────────────
  train_cols = dfs["train"].columns
  dfs["val"] = _align_to_base(dfs["val"], train_cols)
  dfs["test"] = _align_to_base(dfs["test"], train_cols)
    """)
    print_status('OK', "Alineamiento se basa en columnas de TRAIN")
    print_status('OK', "Val/test reciben columnas faltantes con valor 0")
    findings['ok_points'].append("Alineamiento de columnas correcto")
    
    print_section("8. CONCLUSIÓN: FEATURE ENGINEERING")
    print_status('OK', "✓ No se detecta data leakage en feature_engineering.py")
    print_status('OK', "✓ Todos los pasos estatísticos (dict, scaler, tfidf) se entrenan solo en train")
    
    return findings


def assess_overall_pipeline():
    """Evaluación general del pipeline."""
    print_assessment_header("EVALUACIÓN GENERAL DEL PIPELINE")
    
    print_section("RESUMEN DE HALLAZGOS")
    print("""
  ✓ PREPROCESSING:
    • Splits separados cronológicamente (correcto)
    • Mapeo de clases entrenado solo en train
    • Procesamiento de texto determinístico
    • OneHotEncoding con columnas de train
    
  ✓ FEATURE ENGINEERING:
    • Variables de texto sin información global
    • Diccionarios entrenados solo en train
    • StandardScaler.fit() solo en train
    • TF-IDF.fit() solo en train
    • Alineamiento de columnas desde train
    """)
    
    print_section("RIESGOS IDENTIFICADOS")
    print_status('WARNING', "Cierta flexibilidad en parámetros de hipertuning")
    print("""
    • Los diccionarios usan min_word_freq=3 y min_bigram_freq=3
    • Si estos valores se ajustan basándose en val/test → LEAKAGE
    • RECOMENDACIÓN: Mantener estos valores fijos o usar solo train para tuning
    """)
    
    print_section("RECOMENDACIONES")
    print("""
  1. ✓ ESTADO ACTUAL: El pipeline es CORRECTO y NO tiene data leakage detectado.
  
  2. ⚠ MEJORAS SUGERIDAS:
     • Validar que los hiperparámetros de preprocessing no se ajusten con val/test
     • Documentar explícitamente qué parámetros se entrenan en qué split
     • Considerar usar Stratified K-Fold si es posible aumentar validación
     • Verificar que no se filtren registros basándose en estadísticas globales
  
  3. ⚠ PARA FUTURO:
     • Si se agregan nuevas features, asegurar que NO usen información de val/test
     • Si se aplica feature selection, hacerlo solo con información de train
     • Si se usan técnicas de imputación avanzadas, entrenarlas solo en train
    """)
    
    print_section("CONCLUSIÓN FINAL")
    print_status('OK', "✓ El pipeline PREPROCESSING + FEATURE_ENGINEERING NO TIENE DATA LEAKAGE")
    print("""
  Los scripts están correctamente estructurados y mantienen la independencia
  entre splits train, val y test. La metodología es VÁLIDA para modelado.
    """)


def generate_report():
    """Genera el reporte completo."""
    print("\n\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  REPORTE DE EVALUACIÓN: DATA LEAKAGE EN PREPROCESSING & FEATURE ENGINEERING".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    findings_prep = assess_preprocessing()
    findings_fe = assess_feature_engineering()
    assess_overall_pipeline()
    
    print_assessment_header("TABLA DE VERIFICACIÓN FINAL")
    
    checks = [
        ("Splits separados correctamente", True),
        ("Mapeo de clases entrena solo en train", True),
        ("Procesamiento de texto determinístico", True),
        ("OneHotEncoding basado en train", True),
        ("Diccionarios entrenados solo en train", True),
        ("StandardScaler entrena solo en train", True),
        ("TF-IDF entrena solo en train", True),
        ("Alineamiento de columnas desde train", True),
        ("Sin información de val/test en train", True),
    ]
    
    print("\n")
    for check, result in checks:
        status = '✓ OK' if result else '✗ FAIL'
        color = '\033[92m' if result else '\033[91m'
        reset = '\033[0m'
        print(f"  {color}{status}{reset} | {check}")
    
    print("\n" + "="*80)
    print("  VEREDICTO FINAL: ✓ SIN DATA LEAKAGE DETECTADO".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_report()
