import os
import argparse
from datetime import datetime
import re
from difflib import get_close_matches

import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


def _invert_class_mapping(mapping_path):
    if not os.path.exists(mapping_path):
        return {}

    mapping = pd.read_json(mapping_path, typ="series").to_dict()
    # class_mapping.json suele venir como {"label_text": int}
    return {int(v): str(k) for k, v in mapping.items()}


def _get_best_study_name(storage_url, prefix="lightgbm_"):
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    candidates = [s for s in summaries if s.study_name.startswith(prefix) and s.best_trial is not None]
    if not candidates:
        raise ValueError(
            f"No se encontraron estudios con prefijo '{prefix}' y best_trial válido en {storage_url}."
        )
    best = max(candidates, key=lambda s: s.best_trial.value)
    return best.study_name, best.best_trial.value


def _resolve_study_name(storage_url, study_name, prefix="lightgbm_"):
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    names = [s.study_name for s in summaries if s.study_name.startswith(prefix)]
    if not names:
        raise ValueError(f"No hay estudios con prefijo {prefix} en {storage_url}")

    wanted_full = study_name if study_name.startswith(prefix) else f"{prefix}{study_name}"
    if wanted_full in names:
        return wanted_full

    def _norm(x):
        x = x.lower().strip()
        x = re.sub(r"[\s\-]+", "_", x)
        x = re.sub(r"_+", "_", x)
        return x

    wanted_norm = _norm(wanted_full)
    by_norm = {_norm(n): n for n in names}
    if wanted_norm in by_norm:
        return by_norm[wanted_norm]

    matches = get_close_matches(wanted_norm, list(by_norm.keys()), n=1, cutoff=0.55)
    if matches:
        return by_norm[matches[0]]

    raise ValueError(f"No se pudo resolver study '{study_name}'. Disponibles: {', '.join(sorted(names))}")


def _find_model_path(models_dir, short_study_name):
    def _norm(text):
        text = str(text).strip().lower()
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text

    raw = str(short_study_name).strip()
    candidates = [
        raw,
        raw.replace(" ", "_"),
        raw.replace("-", "_"),
        raw.replace(" ", "__"),
        raw.replace("-", ""),
        _norm(raw),
    ]

    # 1) búsqueda exacta por rutas candidatas
    for cand in dict.fromkeys(candidates):
        expected = os.path.join(models_dir, cand, f"model_{cand}.pkl")
        if os.path.exists(expected):
            return expected

    # 2) búsqueda exacta recursiva por nombre de archivo candidato
    for cand in dict.fromkeys(candidates):
        target_name = f"model_{cand}.pkl"
        for root, _, files in os.walk(models_dir):
            if target_name in files:
                return os.path.join(root, target_name)

    # 3) búsqueda flexible por similitud de nombre de carpeta
    folder_names = []
    for name in os.listdir(models_dir):
        abs_name = os.path.join(models_dir, name)
        if os.path.isdir(abs_name):
            folder_names.append(name)

    norm_raw = _norm(raw)
    partial = [
        name for name in folder_names
        if norm_raw in _norm(name) or _norm(name) in norm_raw
    ]
    if len(partial) == 1:
        chosen = partial[0]
        model_candidate = os.path.join(models_dir, chosen, f"model_{chosen}.pkl")
        if os.path.exists(model_candidate):
            return model_candidate
        # fallback dentro de la carpeta elegida
        for root, _, files in os.walk(os.path.join(models_dir, chosen)):
            for fname in files:
                if fname.startswith("model_") and fname.endswith(".pkl"):
                    return os.path.join(root, fname)

    suggestions = ", ".join(sorted(folder_names)) if folder_names else "(sin carpetas de modelos)"
    raise FileNotFoundError(
        f"No se encontró el modelo para study '{short_study_name}'. "
        f"Probé variantes con espacios/guiones. Modelos disponibles: {suggestions}"
    )


def _get_numeric_columns(df):
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "target" in cols:
        cols.remove("target")
    return cols


def _retrain_model_from_optuna(study_full_name, short_study_name, dataset_dir, optuna_db, models_dir):
    train_path = os.path.join(dataset_dir, "df_final_train.csv")
    val_path = os.path.join(dataset_dir, "df_final_val.csv")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Faltan df_final_train.csv o df_final_val.csv para reconstruir el modelo.")

    study = optuna.load_study(study_name=study_full_name, storage=optuna_db)
    if study.best_trial is None:
        raise ValueError(f"El study {study_full_name} no tiene best_trial.")

    df_train = pd.read_csv(train_path, sep=";")
    df_val = pd.read_csv(val_path, sep=";")
    for df in (df_train, df_val):
        df["texto_limpio"] = df["texto_limpio"].fillna("")

    numeric_columns = _get_numeric_columns(df_train)
    pesos_columns = [c for c in numeric_columns if c.startswith("pesos_")]
    numeric_columns = [c for c in numeric_columns if not c.startswith("pesos_")]
    final_columns = numeric_columns + ["texto_limpio"] + pesos_columns

    X_train = df_train[final_columns]
    y_train = df_train["target"]
    X_val = df_val[final_columns]
    y_val = df_val["target"]

    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    best_params = dict(study.best_trial.params)
    tfidf_max_features = best_params.pop("tfidf_max_features", 10000)
    tfidf_min_df = best_params.pop("tfidf_min_df", 2)
    tfidf_max_df = best_params.pop("tfidf_max_df", 0.95)
    tfidf_use_bigrams = best_params.pop("tfidf_use_bigrams", False)
    tfidf_sublinear_tf = best_params.pop("tfidf_sublinear_tf", False)
    tfidf_ngram_range = (1, 2) if tfidf_use_bigrams else (1, 1)

    best_params.setdefault("objective", "multiclass")
    best_params.setdefault("num_class", int(y_full.nunique()))
    best_params.setdefault("n_jobs", -1)
    best_params.setdefault("verbose", -1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_columns),
            ("pesos", "passthrough", pesos_columns),
            (
                "text",
                TfidfVectorizer(
                    max_features=tfidf_max_features,
                    min_df=tfidf_min_df,
                    max_df=tfidf_max_df,
                    ngram_range=tfidf_ngram_range,
                    sublinear_tf=tfidf_sublinear_tf,
                ),
                "texto_limpio",
            ),
        ],
        remainder="drop",
    )

    model = lgb.LGBMClassifier(**best_params)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_full, y_full)

    model_dir = os.path.join(models_dir, short_study_name)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{short_study_name}.pkl")
    joblib.dump(pipeline, model_path)
    return model_path


def _build_labels(y_true, y_pred, inv_mapping):
    labels = sorted(set(pd.Series(y_true).unique()).union(set(pd.Series(y_pred).unique())))
    label_names = [inv_mapping.get(int(lbl), str(lbl)) for lbl in labels]
    return labels, label_names


def _save_confusion_plots(cm, cm_norm, label_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Matriz absoluta
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Matriz de Confusión (conteos)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_ylabel("Clase real")
    ax.set_xlabel("Clase predicha")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_counts.png"), dpi=150)
    plt.close(fig)

    # Matriz normalizada por fila
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Oranges", vmin=0, vmax=1)
    ax.set_title("Matriz de Confusión Normalizada por clase real")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_ylabel("Clase real")
    ax.set_xlabel("Clase predicha")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa en test el mejor modelo LightGBM parametrizado y analiza errores en profundidad."
    )
    parser.add_argument("--dataset-dir", default="./datasets", help="Carpeta de datasets")
    parser.add_argument("--models-dir", default="./models/lgbm", help="Carpeta de modelos LightGBM")
    parser.add_argument("--optuna-db", default="sqlite:///optuna.sqlite3", help="URL de storage de Optuna")
    parser.add_argument("--study-name", default=None, help="Nombre corto del study (sin prefijo lightgbm_)")
    parser.add_argument("--model-path", default=None, help="Path directo al .pkl del modelo")
    parser.add_argument("--output-dir", default="./models/lgbm/evaluation", help="Carpeta para reportes")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    test_path = os.path.join(dataset_dir, "df_final_test.csv")
    class_map_path = os.path.join(dataset_dir, "class_mapping.json")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No existe {test_path}. Ejecuta preprocessing + feature engineering antes.")

    # Resolver estudio / modelo
    if args.model_path:
        model_path = os.path.abspath(args.model_path)
        short_study_name = os.path.splitext(os.path.basename(model_path))[0].replace("model_", "")
        selected_study = f"lightgbm_{short_study_name}"
        best_value = None
    else:
        if args.study_name:
            selected_study = _resolve_study_name(args.optuna_db, args.study_name, prefix="lightgbm_")
            short_study_name = selected_study.replace("lightgbm_", "", 1)
            best_value = None
        else:
            selected_study, best_value = _get_best_study_name(args.optuna_db, prefix="lightgbm_")
            short_study_name = selected_study.replace("lightgbm_", "", 1)

        models_dir_abs = os.path.abspath(args.models_dir)
        try:
            model_path = _find_model_path(models_dir_abs, short_study_name)
        except FileNotFoundError:
            print("No se encontró .pkl del estudio seleccionado. Reconstruyendo modelo desde Optuna...")
            model_path = _retrain_model_from_optuna(
                study_full_name=selected_study,
                short_study_name=short_study_name,
                dataset_dir=dataset_dir,
                optuna_db=args.optuna_db,
                models_dir=models_dir_abs,
            )

    print("=" * 80)
    print(f"Estudio seleccionado: {selected_study}")
    if best_value is not None:
        print(f"Mejor kappa de validación registrado en Optuna: {best_value:.5f}")
    print(f"Modelo cargado desde: {model_path}")
    print("=" * 80)

    # Cargar datos y modelo
    df_test = pd.read_csv(test_path, sep=";")
    if "target" not in df_test.columns:
        raise ValueError("El archivo de test no tiene columna 'target'.")

    y_test = df_test["target"].astype(int)
    X_test = df_test.drop(columns=["target"], errors="ignore")

    model = joblib.load(model_path)

    # Predicción
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).astype(int)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    # Métricas globales
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    inv_mapping = _invert_class_mapping(class_map_path)
    labels, label_names = _build_labels(y_test, y_pred, inv_mapping)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(args.output_dir, f"{short_study_name}_{timestamp}"))
    os.makedirs(out_dir, exist_ok=True)

    _save_confusion_plots(cm, cm_norm, label_names, out_dir)

    # Reporte por clase
    p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({
        "label_id": labels,
        "label_name": label_names,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": s,
    }).sort_values("recall")
    per_class.to_csv(os.path.join(out_dir, "metricas_por_clase.csv"), index=False)

    # Errores en profundidad
    err_df = pd.DataFrame({
        "row_id_test": df_test.index,
        "y_true": y_test.values,
        "y_pred": y_pred.values,
    })
    err_df["true_name"] = err_df["y_true"].map(lambda x: inv_mapping.get(int(x), str(x)))
    err_df["pred_name"] = err_df["y_pred"].map(lambda x: inv_mapping.get(int(x), str(x)))
    err_df["ok"] = (err_df["y_true"] == err_df["y_pred"]).astype(int)

    if y_proba is not None and y_proba.ndim == 2:
        label_to_idx = {int(lbl): i for i, lbl in enumerate(labels) if i < y_proba.shape[1]}
        err_df["proba_pred"] = [
            float(y_proba[i, label_to_idx.get(int(pred), 0)]) if int(pred) in label_to_idx else np.nan
            for i, pred in enumerate(err_df["y_pred"].values)
        ]
        err_df["proba_true"] = [
            float(y_proba[i, label_to_idx.get(int(true), 0)]) if int(true) in label_to_idx else np.nan
            for i, true in enumerate(err_df["y_true"].values)
        ]

    # Dataset completo + predicción para trazabilidad de cada registro
    df_pred = df_test.copy().reset_index(drop=True)
    df_pred.insert(0, "row_id_test", df_test.reset_index().iloc[:, 0].values)
    df_pred["y_true"] = err_df["y_true"].values
    df_pred["y_pred"] = err_df["y_pred"].values
    df_pred["true_name"] = err_df["true_name"].values
    df_pred["pred_name"] = err_df["pred_name"].values
    df_pred["ok"] = err_df["ok"].values
    if "proba_pred" in err_df.columns:
        df_pred["proba_pred"] = err_df["proba_pred"].values
    if "proba_true" in err_df.columns:
        df_pred["proba_true"] = err_df["proba_true"].values

    # Todos los registros evaluados
    df_pred.to_csv(os.path.join(out_dir, "predicciones_detalle_dataset.csv"), index=False)

    # Solo errores, con todas las columnas del dataset
    errores = df_pred[df_pred["ok"] == 0].copy()
    errores.to_csv(os.path.join(out_dir, "errores_detalle_dataset.csv"), index=False)

    # Compatibilidad con reporte anterior
    err_df[err_df["ok"] == 0].copy().to_csv(os.path.join(out_dir, "errores_detalle.csv"), index=False)

    top_conf = (
        errores.groupby(["true_name", "pred_name"]).size().reset_index(name="cantidad")
        .sort_values("cantidad", ascending=False)
    )
    top_conf.to_csv(os.path.join(out_dir, "top_confusiones.csv"), index=False)

    # Matriz en csv para análisis adicional
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix_counts.csv"))
    cm_norm_df = pd.DataFrame(cm_norm, index=label_names, columns=label_names)
    cm_norm_df.to_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"))

    # Resumen en texto
    report_txt = classification_report(y_test, y_pred, labels=labels, target_names=label_names, zero_division=0)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Study: {selected_study}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Kappa: {kappa:.6f}\n\n")
        f.write(report_txt)

    print(f"Accuracy test: {acc:.5f}")
    print(f"Kappa test: {kappa:.5f}")
    print("\nClases con menor recall (más problemáticas):")
    print(per_class[["label_name", "recall", "support"]].head(8).to_string(index=False))

    print("\nTop confusiones (real -> predicha):")
    if not top_conf.empty:
        print(top_conf.head(12).to_string(index=False))
    else:
        print("No hay confusiones: predicción perfecta en test.")

    print("\nArchivos generados:")
    print(f"- {os.path.join(out_dir, 'confusion_matrix_counts.png')}")
    print(f"- {os.path.join(out_dir, 'confusion_matrix_normalized.png')}")
    print(f"- {os.path.join(out_dir, 'metricas_por_clase.csv')}")
    print(f"- {os.path.join(out_dir, 'errores_detalle.csv')}")
    print(f"- {os.path.join(out_dir, 'errores_detalle_dataset.csv')}")
    print(f"- {os.path.join(out_dir, 'predicciones_detalle_dataset.csv')}")
    print(f"- {os.path.join(out_dir, 'top_confusiones.csv')}")
    print(f"- {os.path.join(out_dir, 'classification_report.txt')}")


if __name__ == "__main__":
    main()
