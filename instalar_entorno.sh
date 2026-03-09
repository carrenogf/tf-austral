#!/bin/bash

echo "Iniciando la configuración del entorno definitivo para la Tesis..."

# 1. Crear un entorno nuevo y limpio con Python 3.10
# (Llamaremos a este entorno 'tesis_final')
conda create -n tesis_final python=3.10 -y

# 2. Activar el entorno (usamos 'source' para asegurar compatibilidad en el script)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tesis_final

# 3. Instalar la caja de herramientas base de CUDA vía Conda (Específico para TF 2.13)
echo "Instalando CUDA Toolkit 11.8..."
conda install -c conda-forge cudatoolkit=11.8.0 -y

# 4. Instalar TensorFlow 2.13 y la librería exacta de cuDNN
echo "Instalando TensorFlow y cuDNN..."
python -m pip install tensorflow==2.13.0 nvidia-cudnn-cu11==8.6.0.163

# 5. Configurar las variables de entorno para que TF detecte la GPU automáticamente
echo "Configurando rutas dinámicas de la GPU..."
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib/:\$CUDNN_PATH/lib" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 6. Instalar TODAS las dependencias con las versiones blindadas (Cero conflictos)
echo "Instalando dependencias de ML y NLP..."
python -m pip install \
    "numpy<=1.24.3" \
    "keras<2.14" \
    "typing-extensions<4.6.0" \
    "scikeras<=0.12.0" \
    "spacy<3.8.0" \
    "pydantic<2.0.0" \
    "pandas<2.2.0" \
    "sqlalchemy<2.0" \
    scikit-learn \
    nltk \
    joblib \
    optuna \
    lightgbm \
    optuna-dashboard

# 7. Descargar modelos de lenguaje base para NLP
echo "Descargando modelos de SpaCy y NLTK..."
python -m spacy download es_core_news_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "========================================================"
echo "¡Entorno 'tesis_final' creado y configurado con éxito! 🚀"
echo "Para empezar a trabajar, ejecuta: conda activate tesis_final"
echo "========================================================"
