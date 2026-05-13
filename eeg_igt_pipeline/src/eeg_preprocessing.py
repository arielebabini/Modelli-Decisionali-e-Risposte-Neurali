"""
=============================================================================
EEG Preprocessing Pipeline – Iowa Gambling Task (IGT)
=============================================================================
Dataset: Chávez-Sánchez et al., Scientific Data (2026)
         https://data.mendeley.com/datasets/2pw2m39yct/2

Obiettivo: Classificare decisioni vantaggiose (C/D) vs svantaggiose (A/B)
           usando il segnale EEG nella finestra [-2s, 0s] prima della decisione.

Struttura attesa del dataset:
  root/
  ├── participants.csv
  ├── s-01/
  │   ├── s-01_eeg.csv          (EEG raw, colonne = canali, righe = campioni)
  │   ├── s-01_igt.csv          (dati comportamentali IGT)
  │   └── s-01_processed_eeg.csv  (opzionale)
  ├── s-02/
  │   └── ...
  └── ...

=============================================================================
"""

import os
import re
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
import matplotlib
matplotlib.use("Agg")  # backend non-interattivo per salvataggio figure
import matplotlib.pyplot as plt
from scipy.signal import welch

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COSTANTI GLOBALI
# ---------------------------------------------------------------------------
SFREQ = 256          # Hz – frequenza di campionamento
N_CHANNELS = 21      # canali EEG attivi
TMIN = -2.0          # inizio epoca [s] rispetto all'evento
TMAX = 0.0           # fine epoca [s] rispetto all'evento (momento decisione)
BASELINE = None      # nessuna baseline removal (già inclusa nella finestra)

# Nomi canali secondo sistema 10-20 (paper: 21 elettrodi, ref = Pz)
CH_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz",
    "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4",
    "T4", "T5", "T6", "Pz"
]

# Mapping deck → label
# A, B = disadvantageous (rischiosi, perdite a lungo termine) → label 0
# C, D = advantageous (sicuri, guadagni a lungo termine)    → label 1
DECK_LABEL_MAP = {"A": 0, "B": 0, "C": 1, "D": 1}
LABEL_NAMES = {0: "disadvantageous", 1: "advantageous"}

# Frequenze di filtraggio
BANDPASS_LOW  = 0.5    # Hz
BANDPASS_HIGH = 70.0   # Hz
NOTCH_FREQ    = 50.0   # Hz (Europa) – cambia a 60.0 per standard USA

# ICA
N_ICA_COMPONENTS = 15  # componenti ICA da stimare


# ===========================================================================
# 1. CARICAMENTO DATASET
# ===========================================================================

def discover_subjects(root_dir: str) -> list[str]:
    """
    Scansiona la directory root e restituisce la lista ordinata delle
    directory soggetto (pattern 's-XX').

    Parameters
    ----------
    root_dir : str
        Percorso alla directory radice del dataset.

    Returns
    -------
    list[str]
        Lista di percorsi assoluti alle directory dei soggetti.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory dataset non trovata: {root_dir}")

    subjects = sorted(
        [str(p) for p in root.iterdir()
         if p.is_dir() and re.match(r"s-\d+", p.name)]
    )
    logger.info(f"Trovati {len(subjects)} soggetti in {root_dir}")
    return subjects


def load_eeg_csv(subject_dir: str, subject_id: str) -> Optional[np.ndarray]:
    """
    Carica il file EEG raw da CSV.

    Il file atteso ha:
      - una riga di intestazione con i nomi dei canali
      - N_CHANNELS colonne di valori float (µV)
      - righe = campioni temporali

    Parameters
    ----------
    subject_dir : str
        Directory del soggetto.
    subject_id : str
        Identificativo soggetto (es. 's-01').

    Returns
    -------
    np.ndarray di shape (n_samples, n_channels) oppure None se non trovato.
    """
    path = Path(subject_dir) / f"{subject_id}_eeg.csv"
    if not path.exists():
        # Prova nomi alternativi comuni
        for alt in ["eeg_raw.csv", "EEG.csv", "raw_eeg.csv"]:
            p = Path(subject_dir) / alt
            if p.exists():
                path = p
                break
        else:
            logger.warning(f"File EEG non trovato per {subject_id}")
            return None

    try:
        df = pd.read_csv(path)
        data = df.values.astype(np.float64)
        logger.info(f"  EEG caricato: {data.shape} (campioni × canali)")
        return data
    except Exception as e:
        logger.error(f"  Errore caricamento EEG {subject_id}: {e}")
        return None


def load_igt_csv(subject_dir: str, subject_id: str) -> Optional[pd.DataFrame]:
    """
    Carica il file dei dati comportamentali IGT.

    Colonne attese:
      - TRIAL       : numero di trial (1–200)
      - DECK        : 'A', 'B', 'C' o 'D'
      - EEG_SAMPLE  : indice campione EEG sincronizzato con la decisione
      - WIN         : crediti vinti
      - LOSS        : crediti persi
      - NET         : WIN + LOSS (balance)
      - RT          : reaction time (ms)

    Parameters
    ----------
    subject_dir : str
    subject_id  : str

    Returns
    -------
    pd.DataFrame oppure None.
    """
    path = Path(subject_dir) / f"{subject_id}_igt.csv"
    if not path.exists():
        for alt in ["igt.csv", "IGT.csv", "behavioral.csv"]:
            p = Path(subject_dir) / alt
            if p.exists():
                path = p
                break
        else:
            logger.warning(f"File IGT non trovato per {subject_id}")
            return None

    try:
        df = pd.read_csv(path)
        # Normalizza nomi colonne (maiuscolo, rimuovi spazi)
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        logger.info(f"  IGT caricato: {len(df)} trial, colonne: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"  Errore caricamento IGT {subject_id}: {e}")
        return None


# ===========================================================================
# 2. CREAZIONE OGGETTO MNE RAW
# ===========================================================================

def build_mne_raw(
    eeg_data: np.ndarray,
    ch_names: Optional[list] = None,
    sfreq: float = SFREQ
) -> mne.io.RawArray:
    """
    Crea un oggetto MNE RawArray a partire dai dati EEG raw.

    MNE lavora in Volt: i dati vengono convertiti da µV (CSV) → V.
    La shape attesa da MNE è (n_channels, n_samples), quindi si traspone.

    Parameters
    ----------
    eeg_data : np.ndarray, shape (n_samples, n_channels)
    ch_names : list di stringhe (default: CH_NAMES globale)
    sfreq    : frequenza di campionamento

    Returns
    -------
    mne.io.RawArray
    """
    if ch_names is None:
        ch_names = CH_NAMES

    n_samples, n_ch = eeg_data.shape

    # Adatta CH_NAMES se il numero di canali differisce
    if n_ch != len(ch_names):
        logger.warning(
            f"  Numero canali nel file ({n_ch}) ≠ CH_NAMES ({len(ch_names)}). "
            "Uso nomi generici."
        )
        ch_names = [f"EEG{i:03d}" for i in range(n_ch)]

    # Converti µV → V (standard MNE)
    data_v = eeg_data.T * 1e-6  # shape: (n_ch, n_samples)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg"
    )
    raw = mne.io.RawArray(data_v, info, verbose=False)
    logger.info(f"  MNE RawArray creato: {raw.info['nchan']} canali, "
                f"{raw.n_times} campioni, {raw.times[-1]:.1f}s")
    return raw


# ===========================================================================
# 3. PREPROCESSING EEG
# ===========================================================================

def set_reference(raw: mne.io.BaseRaw, ref: str = "average") -> mne.io.BaseRaw:
    """
    Applica la re-referencing EEG.

    Il paper usa Pz come riferimento durante l'acquisizione; per l'analisi
    è comune usare la media di tutti gli elettrodi (Common Average Reference)
    che minimizza il contributo di artefatti comuni.

    Parameters
    ----------
    raw : MNE Raw
    ref : 'average' (CAR) oppure nome di un canale specifico (es. 'A1')

    Returns
    -------
    MNE Raw con nuovo riferimento.
    """
    if ref == "average":
        raw_ref, _ = mne.set_eeg_reference(raw, ref_channels="average",
                                            copy=True, verbose=False)
        logger.info("  Re-referencing: Common Average Reference (CAR)")
    else:
        raw_ref, _ = mne.set_eeg_reference(raw, ref_channels=[ref],
                                            copy=True, verbose=False)
        logger.info(f"  Re-referencing: {ref}")
    return raw_ref


def apply_filters(
    raw: mne.io.BaseRaw,
    l_freq: float = BANDPASS_LOW,
    h_freq: float = BANDPASS_HIGH,
    notch_freq: float = NOTCH_FREQ
) -> mne.io.BaseRaw:
    """
    Applica filtraggio bandpass e notch.

    Motivazione neuroscientifica:
      - Bandpass 0.5–70 Hz: rimuove deriva DC (< 0.5 Hz) e rumore HF
        preservando le bande neurofisiologiche di interesse:
          δ (1–4), θ (4–8), α (8–13), β (13–30), γ (30–70) Hz
      - Notch 50 Hz: rimuove interferenza rete elettrica (standard europeo).
        Il paper usa un notch a 60 Hz (standard messicano/nordamericano);
        adatta NOTCH_FREQ in base al dataset.

    Parameters
    ----------
    raw        : MNE Raw
    l_freq     : frequenza di taglio bassa del bandpass
    h_freq     : frequenza di taglio alta del bandpass
    notch_freq : frequenza notch (Hz)

    Returns
    -------
    MNE Raw filtrato.
    """
    raw_f = raw.copy()

    # --- Bandpass filter (FIR, metodo default MNE = firwin) ---
    raw_f.filter(
        l_freq=l_freq, h_freq=h_freq,
        method="fir", fir_window="hamming",
        verbose=False
    )
    logger.info(f"  Bandpass FIR: {l_freq}–{h_freq} Hz")

    # --- Notch filter (IIR notch a fase zero) ---
    raw_f.notch_filter(
        freqs=notch_freq,
        method="fir",
        verbose=False
    )
    logger.info(f"  Notch filter: {notch_freq} Hz")

    return raw_f


def run_ica(
    raw: mne.io.BaseRaw,
    n_components: int = N_ICA_COMPONENTS,
    random_state: int = 42
) -> Tuple[mne.io.BaseRaw, ICA]:
    """
    Esegue ICA per rimozione artefatti oculari e muscolari.

    Motivazione neuroscientifica:
      L'ICA (Independent Component Analysis) decompone il segnale EEG in
      sorgenti statisticamente indipendenti. Le componenti associate a:
        - Blink: ampiezza elevata in Fp1, Fp2, Fpz (elettrodi frontali)
        - Movimenti oculari: polarità opposta Fp1 vs Fp2
      vengono identificate e rimosse prima di ricostruire il segnale.

    NOTA: Per un'analisi completamente automatica si usa
          mne.preprocessing.find_eog_events + ICA.find_bads_eog().
          Qui usiamo il metodo automatico 'fastica' con detection
          basata su correlazione con canali EOG frontali (Fp1, Fp2, Fpz).

    Parameters
    ----------
    raw          : MNE Raw (già filtrato)
    n_components : numero di componenti ICA
    random_state : seed per riproducibilità

    Returns
    -------
    (raw_clean, ica) : segnale pulito e oggetto ICA fittato
    """
    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=random_state,
        max_iter=1500,
        fit_params={"tol": 1e-4},
        verbose=False
    )

    # Fitta ICA sul segnale filtrato ≥ 1 Hz (raccomandazione MNE per ICA)
    raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_hp, verbose=False)
    logger.info(f"  ICA fittato ({n_components} componenti)")

    # --- Identificazione automatica artefatti oculari ---
    # Usa i canali frontali come proxy per EOG
    eog_channels = [ch for ch in ["Fp1", "Fp2", "Fpz"] if ch in raw.ch_names]
    excluded = []

    if eog_channels:
        eog_indices, eog_scores = ica.find_bads_eog(
            raw, ch_name=eog_channels,
            threshold=3.0, verbose=False
        )
        ica.exclude = eog_indices
        excluded = eog_indices
        logger.info(f"  Componenti EOG escluse: {eog_indices}")
    else:
        logger.warning("  Nessun canale EOG frontale trovato per detection automatica.")

    # Applica ICA (ricostruzione senza componenti escluse)
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    if not excluded:
        logger.info("  ICA applicata senza esclusioni automatiche.")

    return raw_clean, ica


def normalize_signal(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Normalizzazione z-score canale per canale.

    Motivazione: i segnali fisiologici variano considerevolmente tra soggetti
    (inter-subject variability). La normalizzazione z-score per canale
    riduce questa variabilità e migliora la convergenza dei classificatori ML.

    Formula: z = (x - µ) / σ  per ogni canale

    Parameters
    ----------
    raw : MNE Raw

    Returns
    -------
    MNE Raw normalizzato.
    """
    raw_n = raw.copy()
    data = raw_n.get_data()  # shape: (n_ch, n_samples)
    mu = data.mean(axis=1, keepdims=True)
    sigma = data.std(axis=1, keepdims=True)
    sigma[sigma < 1e-10] = 1e-10  # evita divisione per zero
    raw_n._data = (data - mu) / sigma
    logger.info("  Normalizzazione z-score per canale applicata")
    return raw_n


# ===========================================================================
# 4. SINCRONIZZAZIONE EEG – IGT
# ===========================================================================

def sync_eeg_igt(
    igt_df: pd.DataFrame,
    n_eeg_samples: int
) -> pd.DataFrame:
    """
    Valida e prepara la sincronizzazione EEG–IGT usando la colonna EEG_SAMPLE.

    La colonna EEG_SAMPLE indica l'indice del campione EEG corrispondente
    al momento in cui il partecipante ha effettuato la scelta.
    Conversione in secondi: t = sample / SFREQ

    Aggiunge colonne:
      - EEG_SAMPLE_INT : indice intero campione
      - EEG_TIME_S     : tempo in secondi
      - LABEL          : 0 (A/B disadvantageous) o 1 (C/D advantageous)
      - VALID          : True se il sample consente un'epoca [-2s, 0s]

    Parameters
    ----------
    igt_df         : DataFrame IGT con colonna EEG_SAMPLE e DECK
    n_eeg_samples  : numero totale di campioni EEG disponibili

    Returns
    -------
    pd.DataFrame con colonne aggiunte.
    """
    df = igt_df.copy()

    # Identifica colonna EEG sample (gestisce varianti di nome)
    eeg_col = None
    for col in df.columns:
        if "EEG" in col and "SAMPLE" in col:
            eeg_col = col
            break
    if eeg_col is None:
        raise ValueError("Colonna EEG_SAMPLE non trovata nel file IGT.")

    df["EEG_SAMPLE_INT"] = df[eeg_col].astype(int)
    df["EEG_TIME_S"] = df["EEG_SAMPLE_INT"] / SFREQ

    # Genera label da DECK/DECISION (gestisce varianti di nome del dataset reale)
    deck_col = None
    for candidate in ["DECK", "DECISION", "CARD", "CHOICE"]:
        if candidate in df.columns:
            deck_col = candidate
            break
    if deck_col is None:
        raise ValueError(
            f"Colonna deck non trovata. Colonne disponibili: {list(df.columns)}"
        )
    logger.info(f"  Colonna deck identificata: '{deck_col}'")

    df["LABEL"] = df[deck_col].map(DECK_LABEL_MAP)
    if df["LABEL"].isna().any():
        unknown = df[df["LABEL"].isna()][deck_col].unique()
        logger.warning(f"  Deck sconosciuto/i: {unknown}. Trial ignorati.")
        df = df.dropna(subset=["LABEL"])

    df["LABEL"] = df["LABEL"].astype(int)

    # Validazione: l'epoca [-2s, 0s] deve essere interamente nel segnale
    n_pre = int(abs(TMIN) * SFREQ)   # campioni prima dell'evento
    min_sample = n_pre
    max_sample = n_eeg_samples - 1

    df["VALID"] = (
        (df["EEG_SAMPLE_INT"] >= min_sample) &
        (df["EEG_SAMPLE_INT"] <= max_sample)
    )

    n_valid = df["VALID"].sum()
    n_total = len(df)
    logger.info(f"  Sincronizzazione: {n_valid}/{n_total} trial validi")
    logger.info(
        f"  Distribuzione label: "
        f"advantageous={df[df['VALID']]['LABEL'].sum()}, "
        f"disadvantageous={(df[df['VALID']]['LABEL'] == 0).sum()}"
    )

    return df


# ===========================================================================
# 5. EPOCH EXTRACTION
# ===========================================================================

def extract_epochs(
    raw: mne.io.BaseRaw,
    sync_df: pd.DataFrame,
    tmin: float = TMIN,
    tmax: float = TMAX
) -> Tuple[mne.Epochs, np.ndarray, np.ndarray]:
    """
    Estrae le epoche EEG dalla finestra [tmin, tmax] prima della decisione.

    Motivazione neuroscientifica:
      La finestra [-2s, 0s] cattura l'attività cerebrale durante la fase
      di "pre-decisione" (deliberazione). Studi ERP sul IGT mostrano che
      componenti come FRN (Feedback-Related Negativity) e P300 emergono
      in relazione alla decisione. La finestra pre-decisionale è associata
      all'attività di prefrontal cortex (Fz, F3, F4) legata alla valutazione
      del rischio e alla memoria di lavoro (componente decisionale).

    Parameters
    ----------
    raw      : MNE Raw (preprocessato)
    sync_df  : DataFrame con colonne EEG_SAMPLE_INT, LABEL, VALID
    tmin     : inizio epoca [s]
    tmax     : fine epoca [s]

    Returns
    -------
    (epochs, labels, sample_indices)
    """
    valid_trials = sync_df[sync_df["VALID"]].reset_index(drop=True)

    if len(valid_trials) == 0:
        raise ValueError("Nessun trial valido per estrarre epoche!")

    # Costruisce array di eventi MNE: (sample, 0, event_id)
    # event_id: 1 = disadvantageous, 2 = advantageous
    event_id_map = {0: 1, 1: 2}  # label → MNE event_id
    events = np.zeros((len(valid_trials), 3), dtype=int)
    events[:, 0] = valid_trials["EEG_SAMPLE_INT"].values
    events[:, 2] = valid_trials["LABEL"].map(event_id_map).values

    event_dict = {"disadvantageous": 1, "advantageous": 2}

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=BASELINE,
        preload=True,
        event_repeated="drop",   # se due trial hanno lo stesso campione, tieni il primo
        verbose=False
    )

    labels = valid_trials["LABEL"].values
    sample_indices = valid_trials["EEG_SAMPLE_INT"].values

    logger.info(
        f"  Epoche estratte: {len(epochs)} "
        f"| shape per epoca: {epochs.get_data().shape[1:]}"
    )
    return epochs, labels, sample_indices


# ===========================================================================
# 6. OUTPUT E SALVATAGGIO
# ===========================================================================

def save_subject_output(
    subject_id: str,
    epochs: mne.Epochs,
    labels: np.ndarray,
    sample_indices: np.ndarray,
    output_dir: str
) -> dict:
    """
    Salva le epoche e label di un soggetto in formato NumPy.

    File salvati:
      - {subject_id}_epochs.npy   : shape (n_epochs, n_channels, n_times)
      - {subject_id}_labels.npy   : shape (n_epochs,)
      - {subject_id}_samples.npy  : shape (n_epochs,) indici campione
      - {subject_id}_info.npz     : metadati (ch_names, sfreq, tmin, tmax)

    Parameters
    ----------
    subject_id    : str
    epochs        : MNE Epochs
    labels        : array di label (0/1)
    sample_indices: array di indici campione EEG
    output_dir    : directory di output

    Returns
    -------
    dict con percorsi dei file salvati.
    """
    out = Path(output_dir) / "epochs"
    out.mkdir(parents=True, exist_ok=True)

    X = epochs.get_data()  # (n_epochs, n_ch, n_times)

    np.save(out / f"{subject_id}_epochs.npy", X)
    np.save(out / f"{subject_id}_labels.npy", labels)
    np.save(out / f"{subject_id}_samples.npy", sample_indices)
    np.savez(
        out / f"{subject_id}_info.npz",
        ch_names=epochs.ch_names,
        sfreq=epochs.info["sfreq"],
        tmin=epochs.tmin,
        tmax=epochs.tmax
    )

    paths = {
        "epochs": str(out / f"{subject_id}_epochs.npy"),
        "labels": str(out / f"{subject_id}_labels.npy"),
    }
    logger.info(f"  Salvato: {out}/{subject_id}_*.npy  (shape X={X.shape})")
    return paths


def build_dataset_dataframe(output_dir: str) -> pd.DataFrame:
    """
    Aggrega gli output di tutti i soggetti in un unico DataFrame Pandas.

    Il DataFrame ha le colonne:
      subject_id, epoch_idx, label, label_name, ch_<nome>, t_<idx>
    (formato "flat": ogni riga = un trial, le colonne sono feature EEG
    flatten su tutti i canali × tutti i time points)

    ATTENZIONE: per dataset grandi, preferire il formato .npy per soggetto.

    Returns
    -------
    pd.DataFrame
    """
    epochs_dir = Path(output_dir) / "epochs"
    all_rows = []

    for label_file in sorted(epochs_dir.glob("*_labels.npy")):
        sid = label_file.stem.replace("_labels", "")
        epoch_file = epochs_dir / f"{sid}_epochs.npy"
        if not epoch_file.exists():
            continue

        X = np.load(epoch_file)         # (n_epochs, n_ch, n_times)
        y = np.load(label_file)         # (n_epochs,)

        n_epochs, n_ch, n_times = X.shape
        X_flat = X.reshape(n_epochs, -1)  # (n_epochs, n_ch × n_times)

        for i in range(n_epochs):
            row = {"subject_id": sid, "epoch_idx": i,
                   "label": int(y[i]),
                   "label_name": LABEL_NAMES[int(y[i])]}
            # aggiungi feature piattizzate
            for f_idx, val in enumerate(X_flat[i]):
                row[f"f{f_idx:05d}"] = val
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    logger.info(f"Dataset aggregato: {df.shape[0]} trial, {df.shape[1]} colonne")
    return df


# ===========================================================================
# 7. VISUALIZZAZIONE
# ===========================================================================

def plot_raw_vs_clean(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    subject_id: str,
    output_dir: str,
    duration: float = 5.0
) -> None:
    """
    Confronto visivo EEG prima e dopo preprocessing.
    Mostra i primi `duration` secondi per 6 canali selezionati.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    channels_to_plot = ["Fz", "Cz", "Pz", "F3", "F4", "O1"]
    channels_to_plot = [c for c in channels_to_plot if c in raw_before.ch_names][:6]

    fig, axes = plt.subplots(len(channels_to_plot), 2,
                             figsize=(14, 2.5 * len(channels_to_plot)))
    fig.suptitle(f"EEG Prima / Dopo Preprocessing – {subject_id}", fontsize=13)

    n_pts = int(duration * SFREQ)
    times = np.arange(n_pts) / SFREQ

    for row, ch in enumerate(channels_to_plot):
        idx = raw_before.ch_names.index(ch)
        before = raw_before.get_data()[idx, :n_pts] * 1e6  # → µV
        after  = raw_after.get_data()[idx, :n_pts] * 1e6

        axes[row, 0].plot(times, before, lw=0.7, color="steelblue")
        axes[row, 0].set_ylabel(ch, fontsize=9)
        if row == 0:
            axes[row, 0].set_title("Prima del preprocessing")

        axes[row, 1].plot(times, after, lw=0.7, color="darkorange")
        if row == 0:
            axes[row, 1].set_title("Dopo preprocessing (filtro + ICA + norm)")

    for ax in axes[-1]:
        ax.set_xlabel("Tempo (s)")
    plt.tight_layout()
    fname = fig_dir / f"{subject_id}_raw_vs_clean.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    logger.info(f"  Figura salvata: {fname}")


def plot_psd(
    raw: mne.io.BaseRaw,
    subject_id: str,
    output_dir: str,
    label: str = ""
) -> None:
    """
    Power Spectral Density (Welch) dei canali EEG.
    Utile per verificare la corretta applicazione dei filtri.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    data = raw.get_data() * 1e6  # µV
    fxx, pxx = welch(data, fs=SFREQ, nperseg=SFREQ * 2, axis=1)

    pxx_mean = pxx.mean(axis=0)  # media su canali

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(fxx, pxx_mean, lw=1.5)
    ax.axvline(BANDPASS_LOW,  color="red",    lw=1, ls="--", label=f"BP low {BANDPASS_LOW} Hz")
    ax.axvline(BANDPASS_HIGH, color="green",  lw=1, ls="--", label=f"BP high {BANDPASS_HIGH} Hz")
    ax.axvline(NOTCH_FREQ,    color="purple", lw=1, ls="--", label=f"Notch {NOTCH_FREQ} Hz")
    ax.set_xlabel("Frequenza (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"PSD media canali – {subject_id} {label}")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 80])
    plt.tight_layout()
    fname = fig_dir / f"{subject_id}_psd_{label}.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    logger.info(f"  PSD salvata: {fname}")


def plot_epoch_mean(
    epochs: mne.Epochs,
    subject_id: str,
    output_dir: str
) -> None:
    """
    Traccia la media delle epoche per classe (advantageous vs disadvantageous)
    per un subset di canali frontali.
    """
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    channels = [c for c in ["Fz", "Cz", "F3", "F4"] if c in epochs.ch_names]
    times = epochs.times

    fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 4))
    if len(channels) == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        ch_idx = epochs.ch_names.index(ch)
        for label_id, label_name in [(1, "disadvantageous"), (2, "advantageous")]:
            ep = epochs[label_name].get_data()[:, ch_idx, :] * 1e6
            mean_ep = ep.mean(axis=0)
            sem_ep  = ep.std(axis=0) / np.sqrt(len(ep))
            color = "firebrick" if label_id == 1 else "steelblue"
            axes[i].plot(times, mean_ep, lw=1.5, color=color,
                         label=f"{label_name} (n={len(ep)})")
            axes[i].fill_between(times, mean_ep - sem_ep,
                                  mean_ep + sem_ep, alpha=0.2, color=color)
        axes[i].axvline(0, color="k", lw=1, ls="--", label="Decisione")
        axes[i].set_title(ch)
        axes[i].set_xlabel("Tempo (s)")
        axes[i].set_ylabel("Ampiezza (µV)")
        axes[i].legend(fontsize=7)

    fig.suptitle(f"ERP medio per classe – {subject_id}", fontsize=12)
    plt.tight_layout()
    fname = fig_dir / f"{subject_id}_erp_mean.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    logger.info(f"  ERP medio salvato: {fname}")


# ===========================================================================
# 8. PIPELINE PRINCIPALE
# ===========================================================================

def process_subject(
    subject_dir: str,
    subject_id: str,
    output_dir: str,
    save_figures: bool = True
) -> Optional[dict]:
    """
    Esegue l'intera pipeline di preprocessing per un singolo soggetto.

    Steps:
      1. Carica EEG + IGT
      2. Crea oggetto MNE Raw
      3. Re-referencing (CAR)
      4. Filtraggio bandpass + notch
      5. ICA artifact removal
      6. Normalizzazione z-score
      7. Sincronizzazione EEG–IGT
      8. Estrazione epoche [-2s, 0s]
      9. Salvataggio output
      10. (Opzionale) figure diagnostiche

    Parameters
    ----------
    subject_dir  : str  – directory del soggetto
    subject_id   : str  – ID soggetto
    output_dir   : str  – directory output
    save_figures : bool – se True, genera figure diagnostiche

    Returns
    -------
    dict con statistiche oppure None in caso di errore.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Elaborazione soggetto: {subject_id}")
    logger.info(f"{'='*50}")

    try:
        # --- 1. Caricamento ---
        eeg_data = load_eeg_csv(subject_dir, subject_id)
        igt_df   = load_igt_csv(subject_dir, subject_id)

        if eeg_data is None or igt_df is None:
            return None

        # Verifica shape EEG
        assert eeg_data.ndim == 2, f"EEG shape inattesa: {eeg_data.shape}"
        n_samples, n_ch = eeg_data.shape
        logger.info(f"  EEG shape: {n_samples} campioni × {n_ch} canali "
                    f"({n_samples/SFREQ:.1f}s)")

        # --- 2. MNE Raw ---
        raw = build_mne_raw(eeg_data, sfreq=SFREQ)

        # --- 3. Snapshot pre-processing (per figure) ---
        raw_original = raw.copy() if save_figures else None

        # --- 4. Re-referencing ---
        raw = set_reference(raw, ref="average")

        # --- 5. Filtraggio ---
        if save_figures:
            plot_psd(raw, subject_id, output_dir, label="pre_filter")

        raw = apply_filters(raw)

        if save_figures:
            plot_psd(raw, subject_id, output_dir, label="post_filter")

        # --- 6. ICA ---
        raw, ica = run_ica(raw)

        # --- 7. Normalizzazione ---
        raw = normalize_signal(raw)

        # --- 8. Figure pre/post ---
        if save_figures and raw_original is not None:
            plot_raw_vs_clean(raw_original, raw, subject_id, output_dir)

        # --- 9. Sincronizzazione ---
        sync_df = sync_eeg_igt(igt_df, n_eeg_samples=n_samples)

        # --- 10. Epoche ---
        epochs, labels, samples = extract_epochs(raw, sync_df)

        if save_figures:
            plot_epoch_mean(epochs, subject_id, output_dir)

        # --- 11. Salvataggio ---
        save_subject_output(subject_id, epochs, labels, samples, output_dir)

        stats = {
            "subject_id": subject_id,
            "n_trials_total": len(sync_df),
            "n_epochs": len(labels),
            "n_advantageous": int(labels.sum()),
            "n_disadvantageous": int((labels == 0).sum()),
            "epoch_shape": epochs.get_data().shape,
        }
        return stats

    except Exception as e:
        logger.error(f"  ERRORE per {subject_id}: {e}", exc_info=True)
        return None


def run_full_pipeline(
    dataset_root: str,
    output_dir: str,
    save_figures: bool = True,
    subject_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Esegue la pipeline su tutti i soggetti del dataset.

    Parameters
    ----------
    dataset_root  : str  – directory radice del dataset Mendeley
    output_dir    : str  – directory output
    save_figures  : bool – se True, genera figure diagnostiche per ogni soggetto
    subject_limit : int  – limita l'elaborazione ai primi N soggetti (debug)

    Returns
    -------
    pd.DataFrame con statistiche per soggetto.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    subjects = discover_subjects(dataset_root)
    if subject_limit:
        subjects = subjects[:subject_limit]

    all_stats = []
    for subject_dir in subjects:
        subject_id = Path(subject_dir).name
        stats = process_subject(subject_dir, subject_id, output_dir, save_figures)
        if stats:
            all_stats.append(stats)

    summary_df = pd.DataFrame(all_stats)
    summary_path = Path(output_dir) / "pipeline_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"\n{'='*50}")
    logger.info(f"PIPELINE COMPLETATA: {len(all_stats)}/{len(subjects)} soggetti elaborati")
    logger.info(f"Riepilogo salvato: {summary_path}")
    logger.info(f"{'='*50}")

    return summary_df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EEG Preprocessing Pipeline – IGT Dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Percorso alla directory root del dataset")
    parser.add_argument("--output", type=str, default="./output",
                        help="Directory di output (default: ./output)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Disabilita la generazione di figure diagnostiche")
    parser.add_argument("--limit", type=int, default=None,
                        help="Elabora solo i primi N soggetti (debug)")
    args = parser.parse_args()

    summary = run_full_pipeline(
        dataset_root=args.dataset,
        output_dir=args.output,
        save_figures=not args.no_figures,
        subject_limit=args.limit
    )
    print("\n=== Riepilogo pipeline ===")
    print(summary.to_string(index=False))