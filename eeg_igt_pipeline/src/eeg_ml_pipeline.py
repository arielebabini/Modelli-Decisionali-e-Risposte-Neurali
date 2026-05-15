"""
=============================================================================
EEG Feature Extraction + Machine Learning Pipeline – IGT Dataset
=============================================================================
Dataset : Chávez-Sánchez et al., Scientific Data (2026)
          https://data.mendeley.com/datasets/2pw2m39yct/2

Obiettivo: Classificare decisioni vantaggiose (C/D) vs svantaggiose (A/B)
           dalle feature PSD delle epoche EEG [-2s, 0s] pre-decisione.

Input atteso (output di eeg_preprocessing.py):
  output/
  ├── s-01/
  │   ├── s-01_epochs.npy       (n_epochs, n_channels, n_samples)
  │   ├── s-01_labels.npy       (n_epochs,)  – 0=disadv, 1=adv
  │   └── s-01_samples.npy      (n_epochs,)  – EEG sample index
  ├── s-02/ ...

Uso standalone con dati sintetici:
  python eeg_ml_pipeline.py --mode synthetic

Uso con dati reali (output preprocessing):
  python eeg_ml_pipeline.py --mode real --input ./output --output ./ml_results

=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import os
import warnings
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
)
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COSTANTI GLOBALI
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SFREQ       = 256        # Hz

# Bande di frequenza (Hz): nome → (low, high)
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta" : (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

CH_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz",
    "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4",
    "T4", "T5", "T6", "Pz"
]

# ============================================================================
# SEZIONE 1 – FEATURE EXTRACTION (PSD)
# ============================================================================

def extract_psd_features(
    epoch: np.ndarray,
    sfreq: float = SFREQ,
    freq_bands: Dict[str, Tuple[float, float]] = FREQ_BANDS,
    nperseg: Optional[int] = None
) -> np.ndarray:
    """
    Estrae feature PSD (Welch) per una singola epoca EEG.

    Per ogni canale, calcola la potenza media in ciascuna banda di frequenza.
    Il vettore risultante è:
        [ch0_delta, ch0_theta, ..., ch0_gamma, ch1_delta, ..., chN_gamma]

    Parameters
    ----------
    epoch      : np.ndarray, shape (n_channels, n_samples)
    sfreq      : float  – frequenza di campionamento (Hz)
    freq_bands : dict   – {band_name: (low_hz, high_hz)}
    nperseg    : int    – lunghezza segmento Welch; default=min(256, n_samples)

    Returns
    -------
    np.ndarray, shape (n_channels * n_bands,)
    """
    n_channels, n_samples = epoch.shape

    # Lunghezza segmento Welch: bilancia risoluzione frequenziale e varianza
    if nperseg is None:
        nperseg = min(256, n_samples)

    features = []
    for ch_idx in range(n_channels):
        signal = epoch[ch_idx, :]

        # Welch PSD: freqs in Hz, psd in unità²/Hz
        freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, window="hann")

        for band_name, (low, high) in freq_bands.items():
            # Maschera booleana per la banda corrente
            band_mask = (freqs >= low) & (freqs < high)

            if band_mask.sum() == 0:
                # Banda fuori range (es. sfreq troppo bassa) → 0
                band_power = 0.0
            else:
                # Potenza media nella banda (µV²)
                band_power = np.mean(psd[band_mask])

            features.append(band_power)

    return np.array(features, dtype=np.float32)


def get_feature_names(
    ch_names: List[str] = CH_NAMES,
    freq_bands: Dict[str, Tuple[float, float]] = FREQ_BANDS
) -> List[str]:
    """
    Genera i nomi delle feature nell'ordine prodotto da extract_psd_features.

    Returns
    -------
    list di stringhe tipo ['C3_delta', 'C3_theta', ..., 'Pz_gamma']
    """
    names = []
    for ch in ch_names:
        for band in freq_bands.keys():
            names.append(f"{ch}_{band}")
    return names


def build_feature_matrix(
    epochs_data: np.ndarray,
    sfreq: float = SFREQ,
    freq_bands: Dict[str, Tuple[float, float]] = FREQ_BANDS
) -> Tuple[np.ndarray, List[str]]:
    """
    Costruisce la matrice delle feature per tutte le epoche.

    Parameters
    ----------
    epochs_data : np.ndarray, shape (n_epochs, n_channels, n_samples)
    sfreq       : float
    freq_bands  : dict

    Returns
    -------
    X            : np.ndarray, shape (n_epochs, n_channels * n_bands)
    feature_names: list[str]
    """
    n_epochs, n_channels, n_samples = epochs_data.shape
    logger.info(f"Estrazione PSD: {n_epochs} epoche × {n_channels} canali × "
                f"{len(freq_bands)} bande")

    # Pre-alloca (più efficiente di append)
    n_features = n_channels * len(freq_bands)
    X = np.zeros((n_epochs, n_features), dtype=np.float32)

    for i in range(n_epochs):
        X[i] = extract_psd_features(epochs_data[i], sfreq=sfreq,
                                     freq_bands=freq_bands)

    # Usa i nomi canali reali se disponibili
    if n_channels == len(CH_NAMES):
        feature_names = get_feature_names(CH_NAMES, freq_bands)
    else:
        ch_names_generic = [f"CH{j:02d}" for j in range(n_channels)]
        feature_names = get_feature_names(ch_names_generic, freq_bands)

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, feature_names


# ============================================================================
# SEZIONE 2 – DEFINIZIONE MODELLI ML
# ============================================================================

def get_models(random_state: int = RANDOM_SEED) -> Dict[str, Pipeline]:
    """
    Restituisce un dizionario di Pipeline sklearn pronte all'uso.

    Ogni pipeline include:
      1. StandardScaler  – normalizzazione Z-score (fittat SOLO su train)
      2. Classificatore  – con class_weight='balanced' per gestire sbilanciamento

    I modelli sono:
      - Logistic Regression : lineare, interpretabile, buona baseline
      - SVM (RBF kernel)    : non-lineare, robusto, probability=True per AUC
      - Random Forest       : ensemble, stima importanza feature

    Parameters
    ----------
    random_state : int – seed per riproducibilità

    Returns
    -------
    dict {model_name: sklearn.pipeline.Pipeline}
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",   # gestione sbilanciamento classi
                random_state=random_state,
                solver="lbfgs",
                C=1.0                      # regolarizzazione L2
            ))
        ]),

        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,          # necessario per predict_proba → AUC
                class_weight="balanced",
                random_state=random_state,
                C=1.0,
                gamma="scale"
            ))
        ]),

        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),  # RF non richiede scaling ma lo lasciamo
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,            # alberi non potati (regolarizzato tramite n_min)
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
    }
    return models


# ============================================================================
# SEZIONE 3 – CROSS-VALIDATION (LOSO / GroupKFold)
# ============================================================================

def _run_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    models: Dict[str, Pipeline],
    total_folds: int
) -> Optional[Dict]:
    """
    Esegue un singolo fold per tutti i modelli (helper per parallelizzazione).

    Ogni worker clona i pipeline per evitare race condition.
    """
    from sklearn.base import clone
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, roc_curve)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_subject = np.unique(subject_ids[test_idx])

    if len(np.unique(y_train)) < 2:
        print(f"  [Fold {fold_idx+1}/{total_folds}] SKIP – una sola classe in train "
              f"(sogg. test: {test_subject})")
        return None

    fold_results = {}

    for model_name, pipeline in models.items():
        # clone() crea una copia fresca non fittata – fondamentale per parallelismo
        pipe_clone = clone(pipeline)
        pipe_clone.fit(X_train, y_train)

        y_pred  = pipe_clone.predict(X_test)
        y_proba = pipe_clone.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        if len(np.unique(y_test)) == 2:
            roc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_entry = (fpr, tpr, roc)
        else:
            roc = np.nan
            roc_entry = None

        fold_results[model_name] = {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   roc,
            "roc_entry": roc_entry,
        }

    summary = " | ".join(
        f"{n}: Acc={fold_results[n]['accuracy']:.3f} AUC={fold_results[n]['roc_auc']:.3f}"
        for n in fold_results
    )
    print(f"  [Fold {fold_idx+1:>2}/{total_folds}] Test={test_subject} | {summary}")
    return fold_results


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    models: Dict[str, Pipeline],
    cv_strategy: str = "loso",
    n_splits: int = 5,
    n_jobs: int = -1
) -> Dict[str, Dict]:
    """
    Esegue la cross-validation LOSO o GroupKFold con parallelizzazione joblib.

    GARANZIA ANTI-LEAKAGE:
      - StandardScaler dentro la pipeline → fit solo su train per ogni fold
      - Gruppi (subject_ids) garantiscono separazione per soggetto
      - Ogni fold clona il pipeline → nessuna contaminazione tra processi

    PARALLELIZZAZIONE:
      - joblib.Parallel distribuisce i fold sui core disponibili
      - n_jobs=-1 usa tutti i core (puoi limitare, es. n_jobs=4)
      - Backend 'loky' (default) è process-based, sicuro su Windows

    Parameters
    ----------
    X            : np.ndarray (n_epochs, n_features)
    y            : np.ndarray (n_epochs,)
    subject_ids  : np.ndarray (n_epochs,)
    models       : dict {name: Pipeline}
    cv_strategy  : 'loso' | 'group_kfold'
    n_splits     : k per GroupKFold (ignorato se loso)
    n_jobs       : processi paralleli (-1 = tutti i core)

    Returns
    -------
    results  : dict {model_name: {metric: list_of_fold_scores}}
    roc_data : dict {model_name: [(fpr, tpr, auc), ...]}
    """
    from joblib import Parallel, delayed
    import multiprocessing

    if cv_strategy == "loso":
        cv = LeaveOneGroupOut()
        n_folds_expected = len(np.unique(subject_ids))
        logger.info(f"CV strategy: LOSO ({n_folds_expected} soggetti → {n_folds_expected} fold)")
    else:
        cv = GroupKFold(n_splits=n_splits)
        logger.info(f"CV strategy: GroupKFold (k={n_splits})")

    all_splits = list(cv.split(X, y, groups=subject_ids))
    total_folds = len(all_splits)

    n_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    n_workers = min(n_cores, total_folds)
    logger.info(f"Parallelizzazione: {n_workers} worker su {total_folds} fold")
    logger.info("(I log dei fold arriveranno fuori ordine – normale con parallelismo)\n")

    # Esecuzione parallela: ogni fold è indipendente
    fold_results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_run_single_fold)(
            fold_idx, train_idx, test_idx,
            X, y, subject_ids, models, total_folds
        )
        for fold_idx, (train_idx, test_idx) in enumerate(all_splits)
    )

    # Aggregazione risultati nell'ordine originale dei fold
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results  = {name: {m: [] for m in metric_keys} for name in models}
    roc_data = {name: [] for name in models}

    for fold_result in fold_results_list:
        if fold_result is None:
            continue
        for model_name, metrics in fold_result.items():
            results[model_name]["accuracy"].append(metrics["accuracy"])
            results[model_name]["precision"].append(metrics["precision"])
            results[model_name]["recall"].append(metrics["recall"])
            results[model_name]["f1"].append(metrics["f1"])
            results[model_name]["roc_auc"].append(metrics["roc_auc"])
            if metrics["roc_entry"] is not None:
                roc_data[model_name].append(metrics["roc_entry"])

    return results, roc_data


# ============================================================================
# SEZIONE 4 – METRICHE AGGREGATE
# ============================================================================

def compute_summary_table(
    results: Dict[str, Dict],
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Calcola media ± deviazione standard per ogni metrica e modello.

    Parameters
    ----------
    results    : output di run_cross_validation
    output_dir : se specificato, salva il CSV in questa directory

    Returns
    -------
    pd.DataFrame con colonne MultiIndex (metrica, stat)
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []

    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for m in metric_keys:
            vals = [v for v in metrics[m] if not np.isnan(v)]
            if vals:
                row[f"{m}_mean"] = np.mean(vals)
                row[f"{m}_std"]  = np.std(vals)
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"]  = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    # Tabella formattata per stampa
    print("\n" + "="*80)
    print("RIEPILOGO RISULTATI (media ± std)")
    print("="*80)
    for m in metric_keys:
        col_mean = f"{m}_mean"
        col_std  = f"{m}_std"
        if col_mean in df.columns:
            print(f"\n{m.upper()}")
            for model in df.index:
                mean_val = df.loc[model, col_mean]
                std_val  = df.loc[model, col_std]
                print(f"  {model:<25}: {mean_val:.3f} ± {std_val:.3f}")

    if output_dir:
        path = Path(output_dir) / "results_summary.csv"
        df.to_csv(path)
        logger.info(f"Tabella risultati salvata: {path}")

    return df


# ============================================================================
# SEZIONE 5 – VISUALIZZAZIONI
# ============================================================================

def plot_confusion_matrices(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    models: Dict[str, Pipeline],
    output_dir: str,
    cv_strategy: str = "loso",
    n_splits: int = 5
) -> None:
    """
    Genera e salva la confusion matrix aggregata per ogni modello.

    La CM è la somma di tutte le CM dei fold → rappresenta la performance globale.

    Parameters
    ----------
    X, y, subject_ids : dati
    models            : dict modelli (verranno ri-fittati qui)
    output_dir        : directory dove salvare le figure
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if cv_strategy == "loso":
        cv = LeaveOneGroupOut()
    else:
        cv = GroupKFold(n_splits=n_splits)

    label_names = ["Disadvantageous\n(A/B)", "Advantageous\n(C/D)"]
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    # Accumula CM per fold
    cm_totals = {name: np.zeros((2, 2), dtype=int) for name in models}

    for train_idx, test_idx in cv.split(X, y, groups=subject_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue

        for model_name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            cm_totals[model_name] += confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Plot
    for ax, (model_name, cm) in zip(axes, cm_totals.items()):
        # Normalizza per riga (recall per classe)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names,
            ax=ax, vmin=0, vmax=1,
            annot_kws={"size": 11}
        )
        # Sovrascrivi annotazioni con conteggi assoluti
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.65, f"(n={cm[i,j]})",
                        ha="center", va="center", fontsize=8, color="gray")

        ax.set_title(f"{model_name}\n(normalizzata per riga)", fontsize=10)
        ax.set_xlabel("Predetto", fontsize=9)
        ax.set_ylabel("Vero", fontsize=9)

    plt.suptitle("Confusion Matrix – Aggregata LOSO", fontsize=12, y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices salvate: {path}")


def plot_roc_curves(
    roc_data: Dict[str, List],
    output_dir: str
) -> None:
    """
    Genera la curva ROC media con deviazione standard per ogni modello.

    Interpola le TPR su una griglia comune di FPR [0,1],
    poi calcola media e ±1 std.

    Parameters
    ----------
    roc_data   : output di run_cross_validation {model: [(fpr, tpr, auc_val),...]}
    output_dir : str
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    mean_fpr = np.linspace(0, 1, 200)
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for (model_name, fold_rocs), color in zip(roc_data.items(), colors):
        if not fold_rocs:
            continue

        # Interpola tutte le TPR su griglia comune
        tprs_interp = []
        aucs = []
        for fpr, tpr, auc_val in fold_rocs:
            tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
            tprs_interp[-1][0] = 0.0
            aucs.append(auc_val)

        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs_interp, axis=0)

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        # Curva media
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{model_name} (AUC={mean_auc:.3f} ± {std_auc:.3f})")

        # Banda ±1 std
        ax.fill_between(mean_fpr,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        alpha=0.15, color=color)

    # Linea casuale
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Classificatore casuale (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve – Media ± Std (LOSO)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = Path(output_dir) / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"ROC curves salvate: {path}")


def plot_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    top_n: int = 20
) -> None:
    """
    Addestra Random Forest e Logistic Regression sull'intero dataset
    e visualizza le top-N feature più importanti.

    NB: questo plot è puramente esplicativo (non per evaluation).
        Per evaluation usa sempre la CV.

    Parameters
    ----------
    X, y           : dati completi
    feature_names  : lista nomi feature
    output_dir     : str
    top_n          : int – quante feature mostrare
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Random Forest Importance ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    std_imp = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)

    top_idx_rf = np.argsort(importances)[-top_n:][::-1]
    top_names_rf = [feature_names[i] for i in top_idx_rf]
    top_vals_rf  = importances[top_idx_rf]
    top_std_rf   = std_imp[top_idx_rf]

    axes[0].barh(range(top_n), top_vals_rf[::-1], xerr=top_std_rf[::-1],
                 align="center", color="#4CAF50", alpha=0.8, ecolor="gray")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_names_rf[::-1], fontsize=8)
    axes[0].set_xlabel("Importanza (Gini)", fontsize=10)
    axes[0].set_title(f"Random Forest – Top {top_n} Feature", fontsize=11)
    axes[0].grid(axis="x", alpha=0.3)

    # --- Logistic Regression Coefficients ---
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_SEED, C=1.0
    )
    lr.fit(X_scaled, y)
    coefs = lr.coef_[0]  # shape (n_features,)

    top_idx_lr = np.argsort(np.abs(coefs))[-top_n:][::-1]
    top_names_lr = [feature_names[i] for i in top_idx_lr]
    top_coefs_lr  = coefs[top_idx_lr]

    colors_lr = ["#2196F3" if c > 0 else "#F44336" for c in top_coefs_lr[::-1]]
    axes[1].barh(range(top_n), top_coefs_lr[::-1],
                 align="center", color=colors_lr, alpha=0.8)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_names_lr[::-1], fontsize=8)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Coefficiente (positivo=advantageous)", fontsize=10)
    axes[1].set_title(f"Logistic Regression – Top {top_n} Coeff. (|β|)", fontsize=11)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Importanza Feature EEG per Classificazione IGT", fontsize=13, y=1.01)
    plt.tight_layout()
    path = Path(output_dir) / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance salvata: {path}")

    # Salva ranking CSV
    df_rf = pd.DataFrame({
        "feature": [feature_names[i] for i in top_idx_rf],
        "rf_importance": importances[top_idx_rf],
        "rf_std": std_imp[top_idx_rf],
        "lr_coef": coefs[top_idx_rf]
    })
    csv_path = Path(output_dir) / "feature_ranking.csv"
    df_rf.to_csv(csv_path, index=False)
    logger.info(f"Feature ranking CSV salvato: {csv_path}")


def plot_per_fold_metrics(
    results: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Boxplot delle metriche per fold, per ogni modello.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(18, 5))

    for ax, metric, label in zip(axes, metric_keys, metric_labels):
        data_plot = []
        model_labels = []
        for model_name, metrics in results.items():
            vals = [v for v in metrics[metric] if not np.isnan(v)]
            data_plot.append(vals)
            model_labels.append(model_name.replace(" ", "\n"))

        ax.boxplot(data_plot, labels=model_labels, patch_artist=True,
                   medianprops={"color": "black", "linewidth": 2})
        ax.set_title(label, fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle("Distribuzione Metriche per Fold – LOSO", fontsize=12)
    plt.tight_layout()
    path = Path(output_dir) / "metrics_boxplot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Boxplot metriche salvato: {path}")


# ============================================================================
# SEZIONE 6 – SALVATAGGIO MODELLI
# ============================================================================

def save_models(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    output_dir: str
) -> None:
    """
    Addestra ogni modello sull'intero dataset e lo salva con joblib.

    NB: questi modelli "finali" sono per deployment/inference,
        NON per evaluation (usare sempre la CV per quella).

    Parameters
    ----------
    X, y       : dati completi
    models     : dict {name: Pipeline}
    output_dir : str
    """
    models_dir = Path(output_dir) / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name, pipeline in models.items():
        pipeline.fit(X, y)
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = models_dir / f"{safe_name}.joblib"
        joblib.dump(pipeline, path)
        logger.info(f"Modello salvato: {path}")


# ============================================================================
# SEZIONE 7 – DATI SINTETICI (testing senza dataset reale)
# ============================================================================

def generate_synthetic_data(
    n_subjects: int = 10,
    n_epochs_per_subject: int = 30,
    n_channels: int = 21,
    n_samples: int = 512,   # 2s a 256 Hz
    sfreq: float = SFREQ,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera dati EEG sintetici con struttura realistica per testing.

    Introduce una differenza simulata tra classi:
      - Classe 1 (advantageous): alpha frontale più elevato
      - Classe 0 (disadvantageous): theta frontale più elevato

    Parameters
    ----------
    n_subjects           : int – numero soggetti
    n_epochs_per_subject : int – epoche per soggetto
    n_channels, n_samples, sfreq : parametri EEG

    Returns
    -------
    epochs_data : (n_total_epochs, n_channels, n_samples)
    labels      : (n_total_epochs,)
    subject_ids : (n_total_epochs,)  – int array
    """
    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples) / sfreq

    all_epochs, all_labels, all_subjects = [], [], []

    for subj_idx in range(n_subjects):
        for epoch_idx in range(n_epochs_per_subject):
            label = rng.integers(0, 2)  # 0 o 1
            epoch = np.zeros((n_channels, n_samples))

            for ch in range(n_channels):
                # Pink noise base
                signal = np.zeros(n_samples)
                for freq in np.arange(1, 50, 1.0):
                    amp = 3.0 / freq
                    phase = rng.uniform(0, 2*np.pi)
                    signal += amp * np.sin(2*np.pi*freq*t + phase)
                signal += rng.normal(0, 1.0, n_samples)

                # Differenza di classe: canali frontali (Fz=11, F3=7, F4=8)
                if ch in [7, 8, 11]:
                    if label == 1:
                        # Classe advantageous: alpha (10 Hz) più forte
                        signal += 8.0 * np.sin(2*np.pi*10*t + rng.uniform(0, 2*np.pi))
                    else:
                        # Classe disadvantageous: theta (6 Hz) più forte
                        signal += 8.0 * np.sin(2*np.pi*6*t + rng.uniform(0, 2*np.pi))

                epoch[ch] = signal.astype(np.float32)

            all_epochs.append(epoch)
            all_labels.append(label)
            all_subjects.append(subj_idx)

    return (
        np.array(all_epochs, dtype=np.float32),
        np.array(all_labels, dtype=int),
        np.array(all_subjects, dtype=int)
    )


# ============================================================================
# SEZIONE 8 – CARICAMENTO DATI REALI
# ============================================================================

def load_preprocessed_data(
    input_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica le epoche pre-processate dal disco (output di eeg_preprocessing.py).

    Struttura reale prodotta dal preprocessor:
      input_dir/
        epochs/
          s-01_epochs.npy    (n_epochs, n_channels, n_samples)
          s-01_labels.npy    (n_epochs,)
          s-01_samples.npy   (n_epochs,)   – facoltativo
          s-01_info.npz      – ignorato
          s-02_epochs.npy
          ...

    Se i file sono direttamente in input_dir (senza sottocartella epochs/)
    la funzione li trova comunque automaticamente.

    Returns
    -------
    epochs_data, labels, subject_ids  (arrays concatenati per tutti i soggetti)
    """
    root = Path(input_dir)

    # Cerca i file epochs prima in input_dir/epochs/, poi direttamente in input_dir
    epochs_dir = root / "epochs"
    search_dir = epochs_dir if epochs_dir.is_dir() else root

    # Raccogli tutti i file *_epochs.npy e ordina per ID soggetto
    epoch_files = sorted(search_dir.glob("*_epochs.npy"))

    if not epoch_files:
        raise FileNotFoundError(
            f"Nessun file *_epochs.npy trovato in:\n"
            f"  {search_dir}\n"
            f"Assicurati che il preprocessing sia già stato eseguito."
        )

    logger.info(f"Trovati {len(epoch_files)} soggetti in {search_dir}")

    all_epochs, all_labels, all_subjects = [], [], []

    for subj_idx, ep_path in enumerate(epoch_files):
        # Ricava il prefisso soggetto: "s-01_epochs.npy" → "s-01"
        sid = ep_path.name.replace("_epochs.npy", "")

        labels_path = search_dir / f"{sid}_labels.npy"

        if not labels_path.exists():
            logger.warning(f"  {sid}: file labels mancante ({labels_path.name}), skip.")
            continue

        epochs = np.load(ep_path)        # (n_epochs, n_ch, n_samples)
        labels = np.load(labels_path)    # (n_epochs,)

        # Sanity check shape
        if epochs.ndim != 3:
            logger.warning(f"  {sid}: shape epoche inattesa {epochs.shape}, skip.")
            continue
        if len(epochs) != len(labels):
            logger.warning(
                f"  {sid}: mismatch epoche/labels ({len(epochs)} vs {len(labels)}), skip."
            )
            continue

        all_epochs.append(epochs)
        all_labels.append(labels)
        all_subjects.append(np.full(len(labels), subj_idx, dtype=int))

        logger.info(f"  {sid}: {epochs.shape[0]} epoche  "
                    f"[shape {epochs.shape}]  "
                    f"label dist: {np.bincount(labels.astype(int))}")

    if not all_epochs:
        raise RuntimeError(
            "Nessuna epoca caricata correttamente. "
            "Controlla i file nella directory epochs/."
        )

    epochs_data  = np.concatenate(all_epochs,   axis=0)
    labels_all   = np.concatenate(all_labels,   axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    logger.info(f"\nTotale: {epochs_data.shape[0]} epoche da "
                f"{len(np.unique(subjects_all))} soggetti")
    logger.info(f"Shape array finale: {epochs_data.shape}")
    logger.info(f"Label distribution: {np.bincount(labels_all.astype(int))} "
                f"(0=disadv, 1=adv)")

    return epochs_data, labels_all, subjects_all


# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

def run_pipeline(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    output_dir: str = "./ml_results",
    cv_strategy: str = "loso",
    n_splits: int = 5,
    n_jobs: int = -1,
    save_trained_models: bool = True
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline ML end-to-end.

    Steps:
      1.  Estrazione feature PSD (Welch)
      2.  Definizione modelli
      3.  Cross-validation LOSO / GroupKFold
      4.  Calcolo metriche aggregate
      5.  Visualizzazioni (CM, ROC, Feature Importance, Boxplot)
      6.  Salvataggio risultati CSV + modelli

    Parameters
    ----------
    epochs_data  : (n_epochs, n_channels, n_samples)
    labels       : (n_epochs,)
    subject_ids  : (n_epochs,)
    output_dir   : str
    cv_strategy  : 'loso' | 'group_kfold'
    n_splits     : k per GroupKFold
    n_jobs       : processi paralleli (-1=tutti i core, es. 4 per limitare)
    save_trained_models : bool

    Returns
    -------
    summary_df : pd.DataFrame con media ± std per ogni metrica/modello
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("\n" + "="*60)
    logger.info("EEG ML PIPELINE – AVVIO")
    logger.info("="*60)
    logger.info(f"Epoche: {epochs_data.shape}")
    logger.info(f"Labels: {np.bincount(labels)} (0=disadv, 1=adv)")
    logger.info(f"Soggetti: {len(np.unique(subject_ids))}")
    logger.info(f"Output: {Path(output_dir).resolve()}")

    # ---- Step 1: Feature extraction ---
    logger.info("\n[1/6] Estrazione feature PSD...")
    X, feature_names = build_feature_matrix(epochs_data, sfreq=SFREQ)

    # Salva feature matrix
    feat_path = Path(output_dir) / "feature_matrix.npy"
    np.save(feat_path, X)
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        Path(output_dir) / "feature_names.csv", index=False
    )
    logger.info(f"Feature matrix salvata: {feat_path}")

    # ---- Step 2: Modelli ---
    logger.info("\n[2/6] Inizializzazione modelli...")
    models = get_models(random_state=RANDOM_SEED)

    # ---- Step 3: Cross-validation ---
    logger.info(f"\n[3/6] Cross-validation ({cv_strategy})...")
    results, roc_data = run_cross_validation(
        X, labels, subject_ids, models,
        cv_strategy=cv_strategy, n_splits=n_splits, n_jobs=n_jobs
    )

    # ---- Step 4: Metriche ---
    logger.info("\n[4/6] Calcolo metriche aggregate...")
    summary_df = compute_summary_table(results, output_dir=output_dir)

    # ---- Step 5: Visualizzazioni ---
    logger.info("\n[5/6] Generazione visualizzazioni...")

    # Reset models (re-create to avoid fitted state)
    models_for_plots = get_models(random_state=RANDOM_SEED)

    plot_confusion_matrices(
        X, labels, subject_ids, models_for_plots,
        output_dir=output_dir, cv_strategy=cv_strategy, n_splits=n_splits
    )
    plot_roc_curves(roc_data, output_dir=output_dir)
    plot_feature_importance(X, labels, feature_names, output_dir=output_dir)
    plot_per_fold_metrics(results, output_dir=output_dir)

    # ---- Step 6: Salvataggio modelli ---
    if save_trained_models:
        logger.info("\n[6/6] Salvataggio modelli finali...")
        models_final = get_models(random_state=RANDOM_SEED)
        save_models(X, labels, models_final, output_dir=output_dir)

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETATA")
    logger.info(f"Output in: {Path(output_dir).resolve()}")
    logger.info("="*60)

    return summary_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Feature Extraction + ML Pipeline – IGT Dataset"
    )
    parser.add_argument(
        "--mode", type=str, choices=["synthetic", "real"], default="synthetic",
        help="'synthetic' = dati simulati (default), 'real' = output preprocessing"
    )
    parser.add_argument(
        "--input", type=str, default="./output",
        help="Directory output del preprocessing (solo --mode real)"
    )
    parser.add_argument(
        "--output", type=str, default="./ml_results",
        help="Directory output ML (default: ./ml_results)"
    )
    parser.add_argument(
        "--cv", type=str, choices=["loso", "group_kfold"], default="loso",
        help="Strategia CV: loso (default) oppure group_kfold"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="k per GroupKFold (ignorato con loso)"
    )
    parser.add_argument(
        "--n_subjects", type=int, default=10,
        help="Soggetti sintetici da generare (solo --mode synthetic)"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Processi paralleli per LOSO (-1=tutti i core, es. 4 per limitare)"
    )
    parser.add_argument(
        "--no_save_models", action="store_true",
        help="Non salvare i modelli addestrati"
    )
    args = parser.parse_args()

    # ---- Carica / Genera dati ----
    if args.mode == "synthetic":
        logger.info("Modalità SYNTHETIC: generazione dati EEG simulati...")
        epochs_data, labels, subject_ids = generate_synthetic_data(
            n_subjects=args.n_subjects,
            n_epochs_per_subject=30,
            random_state=RANDOM_SEED
        )
    else:
        logger.info(f"Modalità REAL: caricamento da {args.input}...")
        epochs_data, labels, subject_ids = load_preprocessed_data(args.input)

    # ---- Esegui pipeline ----
    summary = run_pipeline(
        epochs_data=epochs_data,
        labels=labels,
        subject_ids=subject_ids,
        output_dir=args.output,
        cv_strategy=args.cv,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        save_trained_models=not args.no_save_models
    )

    print("\n=== TABELLA FINALE (media ± std) ===")
    # Stampa compatta mean±std per ogni metrica
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    header = f"{'Model':<25}" + "".join(f"  {m[:7]:>9}" for m in metric_keys)
    print(header)
    print("-" * len(header))
    for model in summary.index:
        row = f"{model:<25}"
        for m in metric_keys:
            mean_v = summary.loc[model, f"{m}_mean"]
            std_v  = summary.loc[model, f"{m}_std"]
            row += f"  {mean_v:.3f}±{std_v:.3f}"
        print(row)