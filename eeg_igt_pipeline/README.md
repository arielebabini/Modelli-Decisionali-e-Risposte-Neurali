# EEG Preprocessing Pipeline – Iowa Gambling Task

Pipeline di preprocessing EEG per il progetto universitario *Interfacce Uomo-Macchina*, a.a. 2025/26 – Università degli Studi dell'Insubria.

**Dataset**: Chávez-Sánchez et al., *Scientific Data* (2026) – [Mendeley Data](https://data.mendeley.com/datasets/2pw2m39yct/2)

---

## Struttura del progetto

```
eeg_igt_pipeline/
├── src/
│   ├── eeg_preprocessing.py          ← pipeline principale (modulare)
│   └── generate_synthetic_dataset.py ← generatore dati sintetici (testing)
├── eeg_igt_preprocessing.ipynb       ← notebook interattivo step-by-step
├── requirements.txt
├── data/                             ← (da creare) dataset reale o sintetico
└── output/
    ├── epochs/                       ← .npy per soggetto
    └── figures/                      ← figure diagnostiche
```

---

## Installazione

```bash
pip install -r requirements.txt
```

Dipendenze principali:

| Libreria | Ruolo |
|---|---|
| `mne` ≥ 1.6 | Core EEG: raw, filtering, ICA, epochs |
| `numpy` | Array multidimensionali |
| `pandas` | Gestione dati tabulari (IGT) |
| `scipy` | DSP (Welch PSD) |
| `matplotlib` | Visualizzazioni diagnostiche |
| `scikit-learn` | (downstream) classificatori ML |

---

## Quick start

### Test con dati sintetici (senza dataset reale)

```bash
# Genera 5 soggetti sintetici
python src/generate_synthetic_dataset.py --output ./data/synthetic --n_subjects 5

# Esegui la pipeline
python src/eeg_preprocessing.py \
    --dataset ./data/synthetic \
    --output  ./output \
    --limit   2          # solo 2 soggetti per prova rapida
```

### Con il dataset reale Mendeley

1. Scarica il dataset da https://data.mendeley.com/datasets/2pw2m39yct/2
2. Estrai in `./data/igt_eeg_dataset/`
3. Esegui:

```bash
python src/eeg_preprocessing.py \
    --dataset ./data/igt_eeg_dataset \
    --output  ./output
```

### Notebook interattivo

```bash
jupyter notebook eeg_igt_preprocessing.ipynb
```

---

## Pipeline dettagliata

### Step 1 – Caricamento

La funzione `discover_subjects()` scansiona la directory root cercando cartelle con pattern `s-XX`.
Per ogni soggetto vengono caricati:
- `{sid}_eeg.csv` → matrice `(n_campioni × 21 canali)` in µV
- `{sid}_igt.csv` → DataFrame con colonne `DECK`, `EEG_SAMPLE`, `WIN`, `LOSS`, `RT`

### Step 2 – Re-referencing

Viene applicato il **Common Average Reference (CAR)**: ogni canale viene rireferenziato sottraendo la media istantanea di tutti gli elettrodi. Questo riduce artefatti comuni (es. movimento) e migliora la localizzazione delle sorgenti.

Il paper usa Pz come riferimento di acquisizione; il CAR è la scelta standard per analisi ERP e classificazione.

### Step 3 – Filtraggio

| Filtro | Parametri | Motivazione |
|---|---|---|
| Bandpass FIR | 0.5–70 Hz | Preserva δ/θ/α/β/γ, rimuove deriva DC e HF |
| Notch FIR | 50 Hz (60 Hz USA/MEX) | Rimuove interferenza rete elettrica |

Il paper originale usa un filtro IIR di 6° ordine a 0.5–70 Hz con notch a 60 Hz. La pipeline usa FIR (MNE default) per fase lineare e minore distorsione dei transienti.

### Step 4 – ICA (artifact removal)

L'**Independent Component Analysis** decompone il segnale in sorgenti indipendenti.
Le componenti correlate ai movimenti oculari (blink, saccadi) vengono identificate automaticamente tramite correlazione con i canali frontali `Fp1`, `Fp2`, `Fpz` e rimosse.

Componenti tipicamente escluse:
- **Blink**: ampiezza elevata, simmetrica su Fp1/Fp2, morfologia a campana
- **Movimenti orizzontali**: polarità opposta Fp1 vs Fp2
- Artefatti muscolari (EMG): attività HF > 30 Hz

### Step 5 – Normalizzazione

**Z-score per canale**: `z = (x − µ) / σ`

Riduce la variabilità inter-soggetto (le ampiezze EEG variano di fattore 2–5× tra individui) e migliora la convergenza dei classificatori ML.

### Step 6 – Sincronizzazione EEG–IGT

La colonna `EEG_SAMPLE` nel file IGT indica il campione EEG esatto in cui è avvenuta la decisione.

```
t_decisione [s] = EEG_SAMPLE / 256
```

### Step 7 – Estrazione epoche

Finestra: **[-2s, 0s]** relativa al momento della decisione.

**Motivazione neuroscientifica**: la finestra pre-decisionale cattura:
- **Slow Cortical Potentials** (SCP) in Fz: buildup ~1–2s prima
- **Attività theta frontale** (4–8 Hz): working memory e controllo cognitivo
- **Decision Preceding Negativity** (DPN) in Fz: circa -500ms

Risultato per soggetto: tensore `(n_trial × 21 canali × 512 timepoints)`

### Step 8 – Label generation

| Deck | Tipo | Label |
|---|---|---|
| A, B | Disadvantageous (rischiosi, perdite a lungo termine) | `0` |
| C, D | Advantageous (sicuri, guadagni a lungo termine) | `1` |

### Step 9 – Output

Per ogni soggetto vengono salvati:

```
output/epochs/
├── s-01_epochs.npy    # (n_epochs, 21, 512) float32
├── s-01_labels.npy    # (n_epochs,) int
├── s-01_samples.npy   # (n_epochs,) int – indici campione originali
└── s-01_info.npz      # metadati: ch_names, sfreq, tmin, tmax
```

---

## Caricamento per ML (downstream)

```python
import numpy as np
from pathlib import Path

epochs_dir = Path('./output/epochs')

all_X, all_y, all_subjects = [], [], []
for label_file in sorted(epochs_dir.glob('*_labels.npy')):
    sid = label_file.stem.replace('_labels', '')
    X_s = np.load(epochs_dir / f'{sid}_epochs.npy')  # (n, 21, 512)
    y_s = np.load(label_file)
    all_X.append(X_s)
    all_y.append(y_s)
    all_subjects.extend([sid] * len(y_s))

X = np.concatenate(all_X, axis=0)   # (N_total, 21, 512)
y = np.concatenate(all_y, axis=0)   # (N_total,)
subjects = np.array(all_subjects)   # per cross-validation leave-one-subject-out
```

### Consiglio per la validazione

Dato che i dati fisiologici sono strettamente soggetto-dipendenti, usa la strategia **Leave-One-Subject-Out (LOSO)**:

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups=subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # ... addestra e valuta classificatore
```

---

## Debug e suggerimenti

### Controllo shape dati
```python
# EEG
assert eeg_data.ndim == 2
assert eeg_data.shape[1] == 21, f"Atteso 21 canali, trovato {eeg_data.shape[1]}"

# Epoche
X = epochs.get_data()
assert X.shape[1] == 21       # canali
assert X.shape[2] == 512      # 2s × 256Hz
```

### La pipeline non trova i file EEG/IGT
- Verifica la struttura cartelle: ogni soggetto deve avere una cartella `s-XX/`
- Il codice tenta nomi alternativi (`eeg_raw.csv`, `EEG.csv`, ecc.)
- Controlla con `load_eeg_csv(subject_dir, subject_id)` singolarmente

### L'ICA non rimuove abbastanza artefatti
- Aumenta `N_ICA_COMPONENTS` (default 15)
- Abbassa la soglia in `find_bads_eog(..., threshold=2.5)`
- Per ispezione manuale usa `ica.plot_sources(raw)` nel notebook

### Poche epoche per soggetto
- Verifica che `EEG_SAMPLE` non abbia offset (moltiplicazione per 256 Hz)
- Controlla la colonna con `sync_df[['EEG_SAMPLE_INT', 'EEG_TIME_S', 'VALID']].describe()`

---

## Riferimenti

- Chávez-Sánchez et al. (2026). *Scientific Data*, 13:359. https://doi.org/10.1038/s41597-026-06662-0
- Bechara et al. (1994). *Cognition*, 50, 7–15.
- Cui et al. (2013). *Frontiers in Human Neuroscience*, 7, 776.
- Bianchin & Angrilli (2011). *Brain and Cognition*, 75, 273–280.
