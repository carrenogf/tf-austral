import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import traceback
from sklearn.preprocessing import StandardScaler



def get_modelo_base():
  try:
    
    # leemos
    df = pd.read_csv("./datasets/df_final.csv", sep=';')

    df.drop(columns=['texto_limpio', 'text_size', 'text_words_count'], axis=1, inplace=True)
    df.drop(columns=['description_size', 'description_words_count'], axis=1, inplace=True)
    pesos_cols = [col for col in df.columns if col.startswith('pesos_')]
    df.drop(columns=pesos_cols, axis=1, inplace=True)
    
    return df

  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")
    return None
  



def modelo_text_mining():
  try:
    return  pd.read_csv("./datasets/df_final.csv", sep=';')
  except Exception as e:
    tb = traceback.format_exc()
    print(f"Se produjo un error: {e}")
    print(f"Detalles del error:\n{tb}")
    return None