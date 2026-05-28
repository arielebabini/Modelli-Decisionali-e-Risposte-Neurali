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
import matplotlib.patches as mpatches
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

# Rich – output terminale premium
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box as rbox
from rich.logging import RichHandler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)
logger  = logging.getLogger(__name__)
console = Console()

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

# ---------------------------------------------------------------------------
# Palette colori – stile pubblicazione scientifica
# ---------------------------------------------------------------------------
PALETTE       = ["#1A56A0", "#C0392B", "#1E8449"]   # blu, rosso, verde
PALETTE_LIGHT = ["#AED6F1", "#F1948A", "#A9DFBF"]   # versioni chiare (fill/band)


def _setup_plot_style() -> None:
    """
    Configura lo stile matplotlib per grafici di qualità
    pubblicazione scientifica (light theme pulito, tipografia leggibile).
    Chiamare all'inizio di ogni funzione di plot.
    """
    plt.rcParams.update({
        # Figure
        "figure.facecolor":    "white",
        "figure.edgecolor":    "white",
        # Assi
        "axes.facecolor":      "#F7F9FC",
        "axes.edgecolor":      "#AAAAAA",
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        # Griglia
        "grid.color":          "#DDDDDD",
        "grid.linewidth":      0.6,
        "grid.alpha":          0.9,
        "grid.linestyle":      "--",
        # Font
        "font.family":         "sans-serif",
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.titlepad":       10,
        "axes.labelsize":      11,
        "axes.labelpad":       6,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        # Linee
        "lines.linewidth":     2.2,
        "lines.antialiased":   True,
        # Legenda
        "legend.frameon":      True,
        "legend.framealpha":   0.92,
        "legend.edgecolor":    "#CCCCCC",
        "legend.fontsize":     10,
        "legend.title_fontsize": 10,
        # Salvataggio
        "savefig.dpi":         200,
        "savefig.facecolor":   "white",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.15,
    })


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
        print(f"  \u26a0  [Fold {fold_idx+1}/{total_folds}] SKIP – una sola classe "
              f"in train (sogg. test: {test_subject})")
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

    summary_parts = [
        f"{n}: Acc={fold_results[n]['accuracy']:.3f}  AUC={fold_results[n]['roc_auc']:.3f}"
        for n in fold_results
    ]
    subj_str = "/".join(str(s) for s in test_subject)
    print(f"  \u2714  Fold {fold_idx+1:>2}/{total_folds}  subj={subj_str:>4}  "
          + "  |  ".join(summary_parts))
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
    # Usa sklearn.utils.parallel se disponibile (sklearn >= 1.3, Python 3.11+),
    # altrimenti fallback su joblib (sklearn < 1.3)
    try:
        from sklearn.utils.parallel import Parallel, delayed
    except ImportError:
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
    fold_results_list = Parallel(n_jobs=n_jobs, verbose=0)(
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

    # ---- Tabella Rich per terminale ----
    table = Table(
        title="[bold white]Riepilogo Risultati – Media \u00b1 Std[/bold white]",
        box=rbox.ROUNDED,
        show_header=True,
        header_style="bold white on #1A56A0",
        border_style="#5B8DD9",
        expand=False,
        padding=(0, 1),
        title_style="bold",
    )
    table.add_column("Modello", style="bold", min_width=22, no_wrap=True)
    for ml in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        table.add_column(ml, justify="center", min_width=15)

    def _score_style(v: float) -> str:
        if np.isnan(v): return "dim"
        if v >= 0.75:   return "bold green"
        if v >= 0.60:   return "yellow"
        return "red"

    for model in df.index:
        cells: list = [model]
        for m in metric_keys:
            mean_v = df.loc[model, f"{m}_mean"]
            std_v  = df.loc[model, f"{m}_std"]
            cell   = Text(f"{mean_v:.3f} \u00b1 {std_v:.3f}", style=_score_style(mean_v))
            cells.append(cell)
        table.add_row(*cells)

    console.print()
    console.print(table)
    console.print()

    if output_dir:
        path = Path(output_dir) / "results_summary.csv"
        df.to_csv(path)
        logger.info("  ✓ results_summary.csv")

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
    Stile: light theme scientifico, annotazioni doppie (% + conteggio assoluto).
    """
    _setup_plot_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cv = LeaveOneGroupOut() if cv_strategy == "loso" else GroupKFold(n_splits=n_splits)

    label_names = ["Disadv. (A/B)", "Advan. (C/D)"]
    n_models    = len(models)

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

    # Layout figure
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.8 * n_models, 5.0),
        constrained_layout=True
    )
    if n_models == 1:
        axes = [axes]

    cmap = sns.light_palette("#1A56A0", n_colors=256, as_cmap=True)

    for ax, (model_name, cm) in zip(axes, cm_totals.items()):
        total   = cm.sum()
        acc     = np.trace(cm) / total if total > 0 else 0.0
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        # Heatmap senza annotazioni automatiche
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap=cmap,
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            vmin=0, vmax=1,
            linewidths=2.0,
            linecolor="white",
            cbar_kws={"shrink": 0.72, "label": "Recall per classe"},
        )

        # Annotazioni manuali: % grande + conteggio assoluto piccolo
        for i in range(2):
            for j in range(2):
                pct        = cm_norm[i, j]
                cnt        = cm[i, j]
                text_color = "white" if pct > 0.58 else "#1a1a1a"
                ax.text(j + 0.5, i + 0.38, f"{pct:.1%}",
                        ha="center", va="center",
                        fontsize=16, fontweight="bold", color=text_color)
                ax.text(j + 0.5, i + 0.63, f"n = {cnt}",
                        ha="center", va="center",
                        fontsize=9, color=text_color, alpha=0.80)

        ax.set_title(model_name, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predetto", fontsize=10, labelpad=6)
        ax.set_ylabel("Vero", fontsize=10, labelpad=6)
        ax.tick_params(left=False, bottom=False)

        # Accuracy globale sotto ogni subplot
        ax.text(0.5, -0.16, f"Accuracy globale: {acc:.1%}",
                ha="center", transform=ax.transAxes,
                fontsize=10, color="#444444")

    fig.suptitle(
        "Confusion Matrix – Aggregata per Fold (LOSO)",
        fontsize=14, fontweight="bold", y=1.04
    )
    path = Path(output_dir) / "confusion_matrices.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ confusion_matrices.png")


def plot_roc_curves(
    roc_data: Dict[str, List],
    output_dir: str
) -> None:
    """
    Curva ROC media ± std per ogni modello.
    Stile: light theme scientifico, bande di confidenza, riferimento casuale.
    """
    _setup_plot_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    mean_fpr = np.linspace(0, 1, 300)

    for (model_name, fold_rocs), color, color_light in zip(
        roc_data.items(), PALETTE, PALETTE_LIGHT
    ):
        if not fold_rocs:
            continue

        tprs_interp, aucs = [], []
        for fpr, tpr, auc_val in fold_rocs:
            interped    = np.interp(mean_fpr, fpr, tpr)
            interped[0] = 0.0
            tprs_interp.append(interped)
            aucs.append(auc_val)

        mean_tpr      = np.mean(tprs_interp, axis=0)
        mean_tpr[-1]  = 1.0
        std_tpr       = np.std(tprs_interp, axis=0)
        mean_auc      = np.mean(aucs)
        std_auc       = np.std(aucs)

        # Curva media
        ax.plot(
            mean_fpr, mean_tpr,
            color=color, lw=2.5, zorder=3,
            label=f"{model_name}\n  AUC = {mean_auc:.3f} \u00b1 {std_auc:.3f}"
        )
        # Banda ±1 std
        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            alpha=0.15, color=color, zorder=2
        )
        # Marcatore sul punto mediano
        mid = np.searchsorted(mean_fpr, 0.3)
        ax.plot(mean_fpr[mid], mean_tpr[mid], "o",
                color=color, ms=7, zorder=4, markeredgecolor="white",
                markeredgewidth=0.8)

    # Linea di riferimento casuale
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1.4,
            label="Casuale  (AUC = 0.500)", zorder=1)
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

    ax.set_xlabel("False Positive Rate  (1 \u2013 Specificit\u00e0)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (Sensibilit\u00e0)", fontsize=11)
    ax.set_title("Curva ROC \u2013 Media \u00b1 Std (LOSO)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.legend(
        loc="lower right", fontsize=10,
        title="Modelli", title_fontsize=10,
        framealpha=0.93, edgecolor="#CCCCCC"
    )
    ax.text(0.52, 0.05, "No Skill", color="#AAAAAA",
            fontsize=9, rotation=39, va="bottom")

    path = Path(output_dir) / "roc_curves.png"
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info("  ✓ roc_curves.png")


def plot_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    top_n: int = 20
) -> None:
    """
    Feature importance: Random Forest (Gini) e Logistic Regression (|\u03b2|).
    Stile: light theme scientifico, barre con intensit\u00e0 proporzionale al valore.

    NB: puramente esplicativo (non per evaluation).
    """
    _setup_plot_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    std_imp     = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)

    top_idx_rf   = np.argsort(importances)[-top_n:]          # crescente
    top_names_rf = [feature_names[i] for i in top_idx_rf]   # dal meno al più importante
    top_vals_rf  = importances[top_idx_rf]
    top_std_rf   = std_imp[top_idx_rf]

    # --- Logistic Regression ---
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_SEED, C=1.0
    )
    lr.fit(X_scaled, y)
    coefs = lr.coef_[0]

    top_idx_lr   = np.argsort(np.abs(coefs))[-top_n:]        # crescente
    top_names_lr = [feature_names[i] for i in top_idx_lr]
    top_coefs_lr = coefs[top_idx_lr]

    # ---- Layout ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), constrained_layout=True)

    # --- RF: colormap proporzionale all'importanza ---
    norm_vals = top_vals_rf / (top_vals_rf.max() + 1e-9)
    cmap_rf   = plt.cm.Blues
    colors_rf = [cmap_rf(0.35 + 0.55 * v) for v in norm_vals]

    axes[0].barh(
        range(top_n), top_vals_rf,
        xerr=top_std_rf,
        align="center",
        color=colors_rf,
        edgecolor="white",
        linewidth=0.5,
        error_kw={"elinewidth": 1.2, "ecolor": "#888888", "capsize": 3}
    )
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_names_rf, fontsize=8.5)
    axes[0].set_xlabel("Importanza (Gini medio)", fontsize=10)
    axes[0].set_title(f"Random Forest \u2013 Top {top_n} Feature",
                      fontsize=12, fontweight="bold")
    axes[0].grid(axis="x", alpha=0.5, linestyle="--")
    axes[0].set_axisbelow(True)
    # Valori numerici a destra
    x_offset = top_vals_rf.max() * 0.015
    for i, (val, err) in enumerate(zip(top_vals_rf, top_std_rf)):
        axes[0].text(val + err + x_offset, i, f"{val:.4f}",
                     va="center", fontsize=7.5, color="#444444")

    # --- LR: blu=positivo (advantageous), rosso=negativo ---
    norm_lr   = np.abs(top_coefs_lr) / (np.abs(top_coefs_lr).max() + 1e-9)
    for i, (val, norm) in enumerate(zip(top_coefs_lr, norm_lr)):
        color = PALETTE[0] if val >= 0 else PALETTE[1]
        alpha = 0.45 + 0.50 * float(norm)
        axes[1].barh(i, val, align="center", color=color,
                     alpha=alpha, edgecolor="white", linewidth=0.5)

    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_names_lr, fontsize=8.5)
    axes[1].axvline(0, color="#333333", lw=1.0, zorder=5)
    axes[1].set_xlabel("Coefficiente \u03b2  (positivo \u2192 advantageous)", fontsize=10)
    axes[1].set_title(f"Logistic Regression \u2013 Top {top_n} |\u03b2|",
                      fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.5, linestyle="--")
    axes[1].set_axisbelow(True)

    patch_pos = mpatches.Patch(color=PALETTE[0], alpha=0.85,
                               label="\u2192 Advantageous (C/D)")
    patch_neg = mpatches.Patch(color=PALETTE[1], alpha=0.85,
                               label="\u2192 Disadvantageous (A/B)")
    axes[1].legend(handles=[patch_pos, patch_neg], fontsize=9,
                   loc="lower right", framealpha=0.9)

    fig.suptitle(
        "Importanza Feature EEG \u2013 Classificazione Decisioni IGT",
        fontsize=14, fontweight="bold"
    )
    path = Path(output_dir) / "feature_importance.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ feature_importance.png")

    # Salva ranking CSV
    df_rf = pd.DataFrame({
        "feature":       [feature_names[i] for i in top_idx_rf],
        "rf_importance": importances[top_idx_rf],
        "rf_std":        std_imp[top_idx_rf],
        "lr_coef":       coefs[top_idx_rf]
    })
    csv_path = Path(output_dir) / "feature_ranking.csv"
    df_rf.to_csv(csv_path, index=False)
    logger.info("  ✓ feature_ranking.csv")


def plot_per_fold_metrics(
    results: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Box plot + strip plot delle metriche per fold, per ogni modello.
    Usa automaticamente violin (N>=8 fold) o boxplot (N<8 fold) per evitare
    artefatti KDE con pochi punti. I singoli fold sono sempre visibili.
    """
    _setup_plot_style()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    # Costruisci DataFrame long-form per seaborn
    model_order = list(results.keys())
    rows = []
    for model_name in model_order:
        metrics = results[model_name]
        for metric in metric_keys:
            for val in metrics[metric]:
                if not np.isnan(val):
                    rows.append({"Model": model_name, "Metric": metric, "Value": val})
    df_long = pd.DataFrame(rows)

    palette = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(model_order)}

    # Etichette asse X abbreviate
    short = {
        "Logistic Regression": "Log.\nReg.",
        "SVM (RBF)": "SVM",
        "Random Forest": "Rand.\nForest",
    }
    tick_labels = [short.get(m, m) for m in model_order]

    # Determina se usare violin (N >= 8 per fold) o boxplot (N < 8)
    n_folds = max(
        len([v for v in results[m]["accuracy"] if not np.isnan(v)])
        for m in model_order
    )
    use_violin = n_folds >= 8

    fig, axes = plt.subplots(
        1, len(metric_keys),
        figsize=(4.4 * len(metric_keys), 5.4),
        constrained_layout=True
    )

    for ax, metric, label in zip(axes, metric_keys, metric_labels):
        df_m = df_long[df_long["Metric"] == metric]

        if use_violin:
            # Violin (distribuzione) – solo per N >= 8
            sns.violinplot(
                x="Model", y="Value", data=df_m, ax=ax,
                palette=palette,
                order=model_order,
                inner=None,
                cut=0,
                bw_adjust=0.8,
                linewidth=1.2,
                saturation=0.70,
            )
        else:
            # Boxplot – più robusto per N < 8 (evita artefatti KDE)
            sns.boxplot(
                x="Model", y="Value", data=df_m, ax=ax,
                palette=palette,
                order=model_order,
                width=0.45,
                linewidth=1.4,
                fliersize=0,          # nascondi outlier (mostrati dallo strip)
                saturation=0.72,
                boxprops={"alpha": 0.75},
                medianprops={"color": "#111111", "linewidth": 2.2},
                whiskerprops={"linewidth": 1.2, "linestyle": "--"},
                capprops={"linewidth": 1.4},
            )

        # Strip plot – punti singoli fold (sempre visibile)
        sns.stripplot(
            x="Model", y="Value", data=df_m, ax=ax,
            palette=palette,
            order=model_order,
            size=7,
            alpha=0.85,
            jitter=0.12,
            zorder=4,
            linewidth=0.6,
            edgecolor="white",
        )

        # Se violin: aggiungi mediana come linea nera manuale
        if use_violin:
            for i, name in enumerate(model_order):
                vals = df_m[df_m["Model"] == name]["Value"].dropna()
                if len(vals):
                    med = vals.median()
                    ax.hlines(med, i - 0.22, i + 0.22,
                              colors="#111111", linewidths=2.2, zorder=5)

        # Annotazione n= e media sotto ogni gruppo
        for i, name in enumerate(model_order):
            vals = df_m[df_m["Model"] == name]["Value"].dropna()
            n    = len(vals)
            mean = vals.mean() if n else float("nan")
            ax.text(i, -0.08, f"n={n}\n\u03bc={mean:.2f}",
                    ha="center", va="top", fontsize=7.5,
                    color="#555555", transform=ax.get_xaxis_transform())

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim([-0.02, 1.07])
        ax.set_xlabel("")
        ax.set_ylabel("Score" if metric == "accuracy" else "", fontsize=10)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.axhline(0.5, color="#BBBBBB", lw=0.9, ls=":", zorder=1)   # linea chance
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Etichetta tipo grafico in titolo
    plot_type = "Violin" if use_violin else "Box"
    fig.suptitle(
        f"Distribuzione Metriche per Fold \u2013 LOSO  [{plot_type} plot, n={n_folds} fold]",
        fontsize=13, fontweight="bold"
    )
    path = Path(output_dir) / "metrics_boxplot.png"
    plt.savefig(path, dpi=200)
    plt.close()
    logger.info(f"  ✓ metrics_boxplot.png  [{plot_type} plot, n={n_folds} fold]")


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
        logger.info(f"  ✓ {safe_name}.joblib")


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

    logger.info(f"Trovati {len(epoch_files)} soggetti")

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

        dist = np.bincount(labels.astype(int))
        logger.info(f"  {sid}: {epochs.shape[0]} epoche  "
                    f"(adv={dist[1] if len(dist)>1 else 0}, disadv={dist[0]})")

    if not all_epochs:
        raise RuntimeError(
            "Nessuna epoca caricata correttamente. "
            "Controlla i file nella directory epochs/."
        )

    epochs_data  = np.concatenate(all_epochs,   axis=0)
    labels_all   = np.concatenate(all_labels,   axis=0)
    subjects_all = np.concatenate(all_subjects, axis=0)

    dist_all = np.bincount(labels_all.astype(int))
    logger.info(f"\nTotale: {epochs_data.shape[0]} epoche | "
                f"adv={dist_all[1] if len(dist_all)>1 else 0}, "
                f"disadv={dist_all[0]} | "
                f"{len(np.unique(subjects_all))} soggetti")

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
    dist = np.bincount(labels)
    logger.info(
        f"Epoche: {epochs_data.shape[0]}  |  "
        f"Soggetti: {len(np.unique(subject_ids))}  |  "
        f"adv={dist[1] if len(dist)>1 else 0}, disadv={dist[0]}"
    )

    # ---- Step 1: Feature extraction ---
    logger.info("\n[1/6] Estrazione feature PSD...")
    X, feature_names = build_feature_matrix(epochs_data, sfreq=SFREQ)

    # Salva feature matrix
    feat_path = Path(output_dir) / "feature_matrix.npy"
    np.save(feat_path, X)
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        Path(output_dir) / "feature_names.csv", index=False
    )
    logger.info("  ✓ feature_matrix.npy  feature_names.csv")

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

    logger.info("\nPIPELINE ML COMPLETATA ✓")

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