# EEG Preprocessing Pipeline – Iowa Gambling Task

Pipeline di preprocessing EEG per il progetto universitario *Interfacce Uomo-Macchina*, a.a. 2025/26 – Università degli Studi dell'Insubria.

> **Corso:** Interfacce Uomo-Macchina | **Docente:** Prof.ssa Silvia Corchs  
> **Dataset:** Chávez-Sánchez et al., *Scientific Data* (2026) – [Mendeley Data](https://data.mendeley.com/datasets/2pw2m39yct/2)  
> **Parte del progetto:** Preprocessing EEG + Sincronizzazione + Estrazione epoche

---

## Indice

1. [Requisiti di sistema](#1-requisiti-di-sistema)
2. [Clonare il repository](#2-clonare-il-repository)
3. [Configurare l'ambiente Python](#3-configurare-lambiente-python)
4. [Scaricare il dataset](#4-scaricare-il-dataset)
5. [Struttura del progetto](#5-struttura-del-progetto)
6. [Eseguire la pipeline](#6-eseguire-la-pipeline)
7. [Pipeline dettagliata](#7-pipeline-dettagliata)
8. [Output prodotti](#8-output-prodotti)
9. [Caricamento dati per ML](#9-caricamento-dati-per-ml)
10. [Debug e risoluzione problemi](#10-debug-e-risoluzione-problemi)
11. [Riferimenti](#11-riferimenti)

---

## 1. Requisiti di sistema

| Requisito | Versione minima | Note |
|---|---|---|
| Python | 3.10 | Verificare con `python3 --version` |
| pip | 23.0 | Aggiornare con `pip install --upgrade pip` |
| RAM | 8 GB | Consigliati 16 GB per elaborare tutti i 59 soggetti |
| Spazio disco | ~4 GB | ~3 GB dataset + ~500 MB output |
| OS | macOS / Linux / Windows | Testato su macOS |

---

## 2. Clonare il repository

```bash
git clone https://github.com/<username>/<nome-repo>.git
cd <nome-repo>/eeg_igt_pipeline
```

---

## 3. Configurare l'ambiente Python

È fortemente consigliato usare un **ambiente virtuale** per isolare le dipendenze.

### macOS / Linux

```bash
# Crea l'ambiente virtuale
python3 -m venv venv

# Attivalo
source venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt
```

### Windows

```bash
# Crea l'ambiente virtuale
python -m venv venv

# Attivalo
venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

> ⚠️ **Ogni volta che apri un nuovo terminale** devi riattivare l'ambiente con `source venv/bin/activate` (macOS/Linux) o `venv\Scripts\activate` (Windows).

### Verifica installazione

```bash
python3 -c "import mne; print('MNE version:', mne.__version__)"
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
```

### Dipendenze principali

| Libreria | Versione | Ruolo |
|---|---|---|
| `mne` | ≥ 1.6 | Core EEG: raw, filtering, ICA, epochs |
| `numpy` | ≥ 1.24 | Array multidimensionali |
| `pandas` | ≥ 2.0 | Gestione dati tabulari (IGT) |
| `scipy` | ≥ 1.11 | DSP (Welch PSD) |
| `matplotlib` | ≥ 3.7 | Visualizzazioni diagnostiche |
| `scikit-learn` | ≥ 1.3 | Classificatori ML (fase successiva) |

---

## 4. Scaricare il dataset

Il dataset **non è incluso nel repository** (3 GB). Va scaricato manualmente.

### Passo 1 — Scarica da Mendeley Data

Vai su: **https://data.mendeley.com/datasets/2pw2m39yct/2**

Clicca **"Download All"** (non è necessario creare un account).

### Passo 2 — Posiziona il dataset

Il dataset scaricato si chiama `An electroencephalographic and behavioral dataset/`.  
Spostalo dentro `data/igt_eeg_dataset/` in modo da ottenere questa struttura:

```
eeg_igt_pipeline/
└── data/
    └── igt_eeg_dataset/
        ├── participants.csv
        ├── s-01/
        │   ├── EEG.csv
        │   ├── IGT.csv
        │   └── processed_EEG.csv
        ├── s-02/
        └── s-59/
```

Da terminale (se il dataset è in Downloads):

```bash
mkdir -p ./data/igt_eeg_dataset
mv ~/Downloads/"An electroencephalographic and behavioral dataset"/* ./data/igt_eeg_dataset/
```

### Passo 3 — Verifica

```bash
ls ./data/igt_eeg_dataset/       # deve mostrare s-01, s-02, ..., participants.csv
ls ./data/igt_eeg_dataset/s-01/  # deve mostrare EEG.csv, IGT.csv, processed_EEG.csv
```

---

## 5. Struttura del progetto

```
eeg_igt_pipeline/
├── src/
│   ├── eeg_preprocessing.py          ← pipeline principale (modulare)
│   └── generate_synthetic_dataset.py ← generatore dati sintetici per testing
├── eeg_igt_preprocessing.ipynb       ← notebook interattivo step-by-step
├── requirements.txt                  ← dipendenze Python
├── README.md                         ← questo file
├── .gitignore
├── data/                             ← ⚠️ NON nel repo — da scaricare (vedi §4)
└── output/                           ← ⚠️ NON nel repo — generato dalla pipeline
    ├── epochs/                       ← file .npy per soggetto
    ├── figures/                      ← figure diagnostiche PNG
    └── pipeline_summary.csv
```

---

## 6. Eseguire la pipeline

### Opzione A — Riga di comando (consigliata)

```bash
# Attiva il venv (se non già attivo)
source venv/bin/activate

# Tutti i 59 soggetti (~20-40 minuti)
python3 src/eeg_preprocessing.py \
    --dataset ./data/igt_eeg_dataset \
    --output  ./output

# Solo i primi N soggetti (test rapido)
python3 src/eeg_preprocessing.py \
    --dataset ./data/igt_eeg_dataset \
    --output  ./output \
    --limit   3

# Senza figure (più veloce)
python3 src/eeg_preprocessing.py \
    --dataset ./data/igt_eeg_dataset \
    --output  ./output \
    --no-figures
```

### Opzione B — Test con dati sintetici (senza dataset reale)

```bash
# Genera soggetti sintetici
python3 src/generate_synthetic_dataset.py \
    --output ./data/synthetic \
    --n_subjects 5

# Esegui la pipeline
python3 src/eeg_preprocessing.py \
    --dataset ./data/synthetic \
    --output  ./output \
    --limit   2
```

### Opzione C — Notebook interattivo

```bash
pip install jupyter
jupyter notebook eeg_igt_preprocessing.ipynb
```

Nel notebook modifica la cella di configurazione:
```python
DATASET_ROOT = './data/igt_eeg_dataset'   # dataset reale
OUTPUT_DIR   = './output'
```

### Argomenti della pipeline

| Argomento | Default | Descrizione |
|---|---|---|
| `--dataset` | obbligatorio | Percorso alla directory root del dataset |
| `--output` | `./output` | Directory dove salvare gli output |
| `--no-figures` | False | Disabilita la generazione di figure PNG |
| `--limit` | None | Elabora solo i primi N soggetti |

### Output atteso (esempio 1 soggetto)

```
INFO | Trovati 59 soggetti in ./data/igt_eeg_dataset
INFO | Elaborazione soggetto: s-01
INFO |   EEG caricato: (270336, 21) (campioni × canali)
INFO |   IGT caricato: 200 trial
INFO |   Re-referencing: Common Average Reference (CAR)
INFO |   Bandpass FIR: 0.5–70.0 Hz
INFO |   Notch filter: 50.0 Hz
INFO |   ICA fittato (15 componenti)
INFO |   Componenti EOG escluse: [0, 1]
INFO |   Normalizzazione z-score per canale applicata
INFO |   Sincronizzazione: 200/200 trial validi
INFO |   Epoche estratte: 200 | shape: (21, 513)
INFO |   Salvato: output/epochs/s-01_*.npy  (shape X=(200, 21, 513))
INFO | PIPELINE COMPLETATA: 1/1 soggetti elaborati
```

---

## 7. Pipeline dettagliata

### Step 1 — Caricamento
Scansiona le cartelle `s-XX`, carica `EEG.csv` (µV, 21 canali) e `IGT.csv` (colonne: `decision`, `EEG sample`, `win`, `lose`, `balance`). Gestisce automaticamente varianti nei nomi di colonne e file.

### Step 2 — Re-referencing (CAR)
**Common Average Reference**: ogni canale viene rireferenziato sottraendo la media istantanea di tutti gli elettrodi. Riduce artefatti comuni e la dipendenza dal riferimento di acquisizione originale (Pz).

### Step 3 — Filtraggio

| Filtro | Parametri | Motivazione |
|---|---|---|
| Bandpass FIR | 0.5–70 Hz | Preserva δ/θ/α/β/γ, rimuove deriva DC e rumore HF |
| Notch FIR | 50 Hz | Rimuove interferenza rete elettrica |

> ⚠️ Il dataset è stato acquisito in Messico (rete a 60 Hz). Se la PSD mostra un picco residuo a 60 Hz, modifica `NOTCH_FREQ = 60.0` in `eeg_preprocessing.py`.

### Step 4 — ICA (artifact removal)
FastICA con 15 componenti. Le componenti correlate agli artefatti oculari (blink, movimenti) vengono identificate automaticamente tramite correlazione con `Fp1`, `Fp2`, `Fpz` e rimosse.

### Step 5 — Normalizzazione
**Z-score per canale**: `z = (x − µ) / σ`. Riduce la variabilità inter-soggetto nelle ampiezze EEG.

### Step 6 — Sincronizzazione EEG–IGT
`t_decisione [s] = EEG_SAMPLE / 256 Hz`

### Step 7 — Estrazione epoche
Finestra **[-2s, 0s]** prima della decisione. Cattura Slow Cortical Potentials, theta frontale e Decision Preceding Negativity.  
Risultato: `(200 trial × 21 canali × 513 timepoints)` per soggetto.

### Step 8 — Label generation

| Deck | Classe | Label |
|---|---|---|
| A, B | Disadvantageous (perdite lungo termine) | `0` |
| C, D | Advantageous (guadagni lungo termine) | `1` |

---

## 8. Output prodotti

```
output/
├── pipeline_summary.csv          ← riepilogo per tutti i soggetti
├── epochs/
│   ├── s-01_epochs.npy           ← (200, 21, 513) float
│   ├── s-01_labels.npy           ← (200,) int — 0=disadv, 1=adv
│   ├── s-01_samples.npy          ← (200,) int — indici campione originali
│   ├── s-01_info.npz             ← metadati (ch_names, sfreq, tmin, tmax)
│   └── ...
└── figures/
    ├── s-01_psd_pre_filter.png   ← PSD prima del filtraggio
    ├── s-01_psd_post_filter.png  ← PSD dopo il filtraggio
    ├── s-01_raw_vs_clean.png     ← segnale prima/dopo preprocessing
    └── s-01_erp_mean.png         ← ERP medio per classe
```

---

## 9. Caricamento dati per ML

```python
import numpy as np
from pathlib import Path

epochs_dir = Path('./output/epochs')

all_X, all_y, all_subjects = [], [], []
for label_file in sorted(epochs_dir.glob('*_labels.npy')):
    sid = label_file.stem.replace('_labels', '')
    X_s = np.load(epochs_dir / f'{sid}_epochs.npy')  # (200, 21, 513)
    y_s = np.load(label_file)
    all_X.append(X_s)
    all_y.append(y_s)
    all_subjects.extend([sid] * len(y_s))

X = np.concatenate(all_X, axis=0)   # (N_totale, 21, 513)
y = np.concatenate(all_y, axis=0)   # (N_totale,)
subjects = np.array(all_subjects)   # per LOSO cross-validation
```

### Validazione raccomandata — LOSO

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups=subjects):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # → addestra classificatore su X_train, valuta su X_test
```

---

## 10. Debug e risoluzione problemi

### La pipeline non trova i file
- Verifica che le cartelle si chiamino `s-XX` (es. `s-01`, `s-10`)
- Controlla: `ls ./data/igt_eeg_dataset/s-01/`

### Errore "Colonna DECISION non trovata"
- Controlla i nomi colonne: `head -2 ./data/igt_eeg_dataset/s-01/IGT.csv`

### Warning "FastICA did not converge"
- Non blocca la pipeline. Aumenta `max_iter` in `eeg_preprocessing.py` se necessario.

### Picco a 60 Hz nel segnale
- Modifica `NOTCH_FREQ = 60.0` in `eeg_preprocessing.py`

### Controllo shape dati
```python
X = np.load('output/epochs/s-01_epochs.npy')
print(X.shape)   # atteso: (200, 21, 513)
```

---

## 11. Riferimenti

- Chávez-Sánchez et al. (2026). *Scientific Data*, 13:359. https://doi.org/10.1038/s41597-026-06662-0
- Bechara et al. (1994). *Cognition*, 50, 7–15.
- Cui et al. (2013). *Frontiers in Human Neuroscience*, 7, 776.
- Bianchin & Angrilli (2011). *Brain and Cognition*, 75, 273–280.
- Gramfort et al. (2013). *Frontiers in Neuroscience*, 7, 267. (MNE-Python)
