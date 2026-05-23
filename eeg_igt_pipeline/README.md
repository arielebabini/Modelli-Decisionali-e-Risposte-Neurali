# EEG Machine Learning Pipeline – Iowa Gambling Task

Pipeline di feature extraction e classificazione ML per il progetto universitario *Interfacce Uomo-Macchina*, a.a. 2025/26 – Università degli Studi dell'Insubria.

> **Corso:** Interfacce Uomo-Macchina | **Docente:** Prof.ssa Silvia Corchs
> **Dataset:** Chávez-Sánchez et al., *Scientific Data* (2026) – [Mendeley Data](https://data.mendeley.com/datasets/2pw2m39yct/2)
> **Parte del progetto:** Feature Extraction PSD + Machine Learning + Validazione LOSO

> ⚠️ **Prerequisito:** questa pipeline richiede l'output della Parte I (preprocessing EEG). Assicurarsi di aver eseguito `eeg_preprocessing.py` e che la directory `output/epochs/` contenga i file `.npy` per ogni soggetto.

---

## Indice

1. [Requisiti di sistema](#1-requisiti-di-sistema)
2. [Struttura del progetto](#2-struttura-del-progetto)
3. [Input atteso](#3-input-atteso)
4. [Eseguire la pipeline](#4-eseguire-la-pipeline)
5. [Pipeline dettagliata](#5-pipeline-dettagliata)
6. [Output prodotti](#6-output-prodotti)
7. [Risultati ottenuti](#7-risultati-ottenuti)
8. [Debug e risoluzione problemi](#8-debug-e-risoluzione-problemi)
9. [Riferimenti](#9-riferimenti)

---

## 1. Requisiti di sistema

| Requisito | Versione minima | Note |
|---|---|---|
| Python | 3.10 | Verificare con `python3 --version` |
| pip | 23.0 | Aggiornare con `pip install --upgrade pip` |
| RAM | 8 GB | Consigliati 16 GB per LOSO completa su 55 soggetti |
| Spazio disco | ~500 MB | Output ML (CSV, PNG, modelli .joblib) |
| CPU | Multi-core | La parallelizzazione LOSO sfrutta tutti i core disponibili |
| OS | macOS / Linux / Windows | Testato su Windows e macOS |

### Dipendenze principali

| Libreria | Versione | Ruolo |
|---|---|---|
| `numpy` | ≥ 1.24 | Array multidimensionali, operazioni vettoriali |
| `scipy` | ≥ 1.10 | Calcolo PSD (`scipy.signal.welch`) |
| `pandas` | ≥ 2.0 | Tabelle risultati, salvataggio CSV |
| `scikit-learn` | ≥ 1.3 | Modelli ML, Pipeline, LOSO, metriche |
| `joblib` | ≥ 1.3 | Parallelizzazione fold LOSO |
| `matplotlib` | ≥ 3.7 | Generazione grafici (ROC, CM, boxplot) |
| `seaborn` | ≥ 0.12 | Heatmap Confusion Matrix |
| `mne` | ≥ 1.6 | Compatibilità formato epoche (Parte I) |

Installare tutte le dipendenze con:

```bash
pip install -r requirements.txt
```

---

## 2. Struttura del progetto

```
eeg_igt_pipeline/
├── src/
│   ├── eeg_preprocessing.py          ← Parte I: preprocessing EEG
│   ├── eeg_ml_pipeline.py            ← Parte II: feature extraction + ML  ← QUI
│   └── generate_synthetic_dataset.py ← generatore dati sintetici per testing
├── requirements.txt
├── README.md                         ← README preprocessing (Parte I)
├── README_ML.md                      ← questo file
├── output/                           ← generato dalla Parte I
│   ├── epochs/
│   │   ├── s-01_epochs.npy           ← input per questa pipeline
│   │   ├── s-01_labels.npy
│   │   ├── s-01_info.npz
│   │   └── ...
│   └── figures/
└── ml_results/                       ← generato da questa pipeline
    ├── results_summary.csv
    ├── feature_matrix.npy
    ├── feature_names.csv
    ├── feature_ranking.csv
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── feature_importance.png
    ├── metrics_boxplot.png
    └── saved_models/
        ├── logistic_regression.joblib
        ├── svm_rbf.joblib
        └── random_forest.joblib
```

---

## 3. Input atteso

La pipeline legge automaticamente i file `.npy` prodotti dalla Parte I nella directory `output/epochs/`:

| File | Shape | Contenuto |
|---|---|---|
| `s-XX_epochs.npy` | `(200, 21, 513)` | Segnale EEG preprocessato per ogni trial |
| `s-XX_labels.npy` | `(200,)` | Label binarie: `0`=svantaggioso (A/B), `1`=vantaggioso (C/D) |
| `s-XX_samples.npy` | `(200,)` | Indici campione EEG originali (opzionale) |
| `s-XX_info.npz` | — | Metadati: `ch_names`, `sfreq`, `tmin`, `tmax` |

La pipeline cerca i file in `input_dir/epochs/` e, se non trovata, direttamente in `input_dir/`.

### Verifica input

```bash
# Controlla che i file siano presenti
ls output/epochs/ | head -10

# Verifica la shape di un file
python3 -c "import numpy as np; X=np.load('output/epochs/s-01_epochs.npy'); print('Shape:', X.shape)"
# atteso: Shape: (200, 21, 513)
```

---

## 4. Eseguire la pipeline

### Modalità LOSO completa (raccomandata per risultati pubblicabili)

La LOSO (Leave-One-Subject-Out) è la strategia di validazione più rigorosa: produce 55 fold, uno per soggetto, garantendo che nessun soggetto appaia sia in train che in test.

#### macOS / Linux

```bash
# Attiva il venv (se non già attivo)
source venv/bin/activate

# LOSO completa – tutti i core disponibili
python3 src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_loso \
    --cv loso

# LOSO completa – core limitati (per non bloccare il PC)
python3 src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_loso \
    --cv loso \
    --n_jobs 4

# LOSO completa – senza salvare i modelli (solo risultati)
python3 src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_loso \
    --cv loso \
    --no_save_models
```

#### Windows CMD

```bash
# LOSO completa – tutti i core disponibili
python src\eeg_ml_pipeline.py ^
    --mode real ^
    --input .\output ^
    --output .\ml_results_loso ^
    --cv loso

# LOSO completa – core limitati
python src\eeg_ml_pipeline.py ^
    --mode real ^
    --input .\output ^
    --output .\ml_results_loso ^
    --cv loso ^
    --n_jobs 4

# LOSO completa – senza salvare i modelli
python src\eeg_ml_pipeline.py ^
    --mode real ^
    --input .\output ^
    --output .\ml_results_loso ^
    --cv loso ^
    --no_save_models
```

> ⏱️ **Tempo stimato:** 15–30 minuti con tutti i core su CPU consumer (55 fold × 3 modelli). Con `--n_jobs 4` stimare 20–40 minuti.

---

### Modalità GroupKFold (test rapido)

La GroupKFold con `k` fold è più veloce della LOSO pur mantenendo la separazione per soggetto. Raccomandata per verificare rapidamente il funzionamento della pipeline.

#### macOS / Linux

```bash
# GroupKFold k=5 (bilanciamento velocità/qualità)
python3 src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_kfold \
    --cv group_kfold \
    --n_splits 5

# GroupKFold k=10 (maggiore stabilità delle stime)
python3 src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_kfold10 \
    --cv group_kfold \
    --n_splits 10

# GroupKFold k=2 (verifica rapida caricamento)
python3 src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_test \
    --cv group_kfold \
    --n_splits 2
```

#### Windows CMD

```bash
# GroupKFold k=5
python src\eeg_ml_pipeline.py ^
    --mode real ^
    --input .\output ^
    --output .\ml_results_kfold ^
    --cv group_kfold ^
    --n_splits 5

# GroupKFold k=10
python src\eeg_ml_pipeline.py ^
    --mode real ^
    --input .\output ^
    --output .\ml_results_kfold10 ^
    --cv group_kfold ^
    --n_splits 10

# GroupKFold k=2
python src\eeg_ml_pipeline.py ^
    --mode real ^
    --input .\output ^
    --output .\ml_results_test ^
    --cv group_kfold ^
    --n_splits 2
```

---

### Modalità dati sintetici (nessun dataset richiesto)

Per testare la pipeline senza il dataset reale, è disponibile la modalità sintetica che genera dati EEG simulati con struttura realistica:

#### macOS / Linux

```bash
python3 src/eeg_ml_pipeline.py \
    --mode synthetic \
    --n_subjects 10 \
    --output ./ml_results_synthetic
```

#### Windows CMD

```bash
python src\eeg_ml_pipeline.py ^
    --mode synthetic ^
    --n_subjects 10 ^
    --output .\ml_results_synthetic
```

---

### Riepilogo argomenti

| Argomento | Tipo | Descrizione | Default |
|---|---|---|---|
| `--mode` | str | `real` o `synthetic` | `synthetic` |
| `--input` | str | Directory output preprocessing (solo `--mode real`) | `./output` |
| `--output` | str | Directory output risultati ML | `./ml_results` |
| `--cv` | str | Strategia CV: `loso` o `group_kfold` | `loso` |
| `--n_splits` | int | k per GroupKFold (ignorato con `loso`) | `5` |
| `--n_jobs` | int | Processi paralleli (`-1` = tutti i core) | `-1` |
| `--n_subjects` | int | Soggetti sintetici (solo `--mode synthetic`) | `10` |
| `--no_save_models` | flag | Non salvare i modelli `.joblib` | disattivo |

---

## 5. Pipeline dettagliata

### Step 1 — Caricamento dati

Scansiona `input_dir/epochs/` cercando tutti i file `*_epochs.npy` e i corrispondenti `*_labels.npy`. Assegna automaticamente un ID soggetto numerico a ogni file per la cross-validation.

```
INFO | Trovati 55 soggetti in output/epochs
INFO |   s-01: 200 epoche  [shape (200, 21, 513)]  label dist: [99 101]
INFO |   s-02: 200 epoche  [shape (200, 21, 513)]  label dist: [112  88]
...
INFO | Totale: 11000 epoche da 55 soggetti
```

### Step 2 — Feature Extraction PSD (Welch)

Per ogni epoca e per ogni canale, viene calcolata la Power Spectral Density con il metodo di Welch (`nperseg=256`, finestra di Hamming) e estratta la potenza media in cinque bande:

| Banda | Range (Hz) | Correlato EEG-IGT |
|---|---|---|
| Delta (δ) | 0.5 – 4 | Processi decisionali lenti |
| Theta (θ) | 4 – 8 | Working memory, FRN (Fz) |
| Alpha (α) | 8 – 13 | Inibizione cognitiva, P300 |
| Beta (β) | 13 – 30 | Preparazione motoria |
| Gamma (γ) | 30 – 45 | Binding cognitivo, N400 |

Risultato: matrice `X` di shape `(11000, 105)` — 21 canali × 5 bande.

### Step 3 — Modelli ML

Tre classificatori in sklearn Pipeline (StandardScaler + classificatore):

| Modello | Caratteristica principale | Parametri chiave |
|---|---|---|
| Logistic Regression | Baseline lineare, interpretabile | `C=1.0`, `solver=lbfgs` |
| SVM (RBF) | Non-lineare, massima AUC | `C=1.0`, `gamma=scale` |
| Random Forest | Ensemble, feature importance | `n_estimators=200` |

Tutti usano `class_weight='balanced'` per gestire lo sbilanciamento per soggetto.

> ⚠️ **Anti-leakage:** lo StandardScaler è **dentro** la Pipeline sklearn, quindi viene fittato esclusivamente sui dati di training di ogni fold.

### Step 4 — Cross-validation LOSO / GroupKFold

La LOSO usa `LeaveOneGroupOut` con `groups=subject_ids`: nessun soggetto appare sia in train che in test nello stesso fold. I fold vengono eseguiti **in parallelo** tramite `joblib.Parallel` con backend `loky` (process-based, sicuro su Windows).

```
INFO | CV strategy: LOSO (55 soggetti → 55 fold)
INFO | Parallelizzazione: 8 worker su 55 fold

  [Fold  1/55] Test=[0] | LR: Acc=0.510 AUC=0.541 | SVM: Acc=0.525 AUC=0.558 | RF: Acc=0.500 AUC=0.531
  [Fold  2/55] Test=[1] | LR: Acc=0.530 AUC=0.551 | ...
  ...
```

### Step 5 — Metriche e visualizzazioni

Per ogni fold vengono calcolate: accuracy, precision, recall, F1-score, ROC-AUC. Al termine vengono prodotti: tabella media ± std, confusion matrix aggregata, ROC curve media, feature importance plot, boxplot per fold.

---

## 6. Output prodotti

```
ml_results/
├── results_summary.csv          ← tabella media ± std per ogni modello e metrica
├── feature_matrix.npy           ← matrice feature X shape (11000, 105)
├── feature_names.csv            ← nomi delle 105 feature (es. Fz_theta, O1_alpha)
├── feature_ranking.csv          ← ranking feature: importanza RF + coefficienti LR
├── confusion_matrices.png       ← CM normalizzata aggregata per i 3 modelli
├── roc_curves.png               ← ROC media ± std per fold, per ogni modello
├── feature_importance.png       ← top-20 feature: importanza RF e coefficienti LR
├── metrics_boxplot.png          ← distribuzione metriche per fold (boxplot)
└── saved_models/
    ├── logistic_regression.joblib   ← modello finale addestrato su tutto il dataset
    ├── svm_rbf.joblib
    └── random_forest.joblib
```

### Formato `results_summary.csv`

```
Model,accuracy_mean,accuracy_std,precision_mean,...,roc_auc_mean,roc_auc_std
Logistic Regression,0.521,0.098,0.529,...,0.545,0.063
SVM (RBF),0.520,0.099,0.534,...,0.553,0.068
Random Forest,0.513,0.084,0.529,...,0.532,0.059
```

---

## 7. Risultati ottenuti

Risultati della cross-validation LOSO su 55 soggetti (11.000 trial totali):

| Modello | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.521 ± 0.098 | 0.529 ± 0.179 | 0.520 ± 0.277 | 0.475 ± 0.188 | **0.545 ± 0.063** |
| SVM (RBF) | 0.520 ± 0.099 | 0.534 ± 0.165 | 0.451 ± 0.289 | 0.431 ± 0.199 | **0.553 ± 0.068** |
| Random Forest | 0.513 ± 0.084 | 0.529 ± 0.133 | 0.546 ± 0.259 | 0.492 ± 0.159 | **0.532 ± 0.059** |

Tutti i modelli superano il caso fortuito (AUC > 0.5). I risultati sono coerenti con la letteratura EEG-IGT cross-subject (Balconi & Angioletti, 2022: AUC ~0.55–0.62 con setting analogo).

Le feature più informative identificate sono: **theta e alpha frontale** su F7, F8, F4, Fz — coerenti con le componenti ERP (FRN, P300) identificate dal paper originale.

---

## 8. Debug e risoluzione problemi

### `FileNotFoundError: Nessun file *_epochs.npy trovato`

Verificare che il preprocessing della Parte I sia stato completato:

```bash
ls output/epochs/*.npy | head -5
# atteso: output/epochs/s-01_epochs.npy, s-01_labels.npy, ...
```

Se la directory è vuota, eseguire prima `eeg_preprocessing.py` (vedi README principale).

### La pipeline impiega troppo tempo

Usare GroupKFold per test rapidi:

```bash
# macOS / Linux
python3 src/eeg_ml_pipeline.py --mode real --input ./output --cv group_kfold --n_splits 2

# Windows
python src\eeg_ml_pipeline.py --mode real --input .\output --cv group_kfold --n_splits 2
```

Oppure limitare i core per non saturare la CPU:

```bash
python3 src/eeg_ml_pipeline.py --mode real --input ./output --cv loso --n_jobs 4
```

### Warning `ConvergenceWarning` da Logistic Regression

Non blocca la pipeline. Se si vuole eliminare, aumentare `max_iter` nella funzione `get_models()` in `eeg_ml_pipeline.py`:

```python
LogisticRegression(max_iter=2000, ...)
```

### `ValueError: Only one class present in y_true`

Alcuni soggetti hanno una distribuzione molto sbilanciata (es. s-51: 167 vantaggiosi su 200). La pipeline gestisce automaticamente questo caso escludendo il fold dal calcolo dell'AUC e registrando un warning. Non richiede intervento manuale.

### I log dei fold arrivano fuori ordine

Comportamento normale con la parallelizzazione `joblib`. I risultati finali sono comunque aggregati nell'ordine corretto.

### Verifica shape feature matrix

```python
import numpy as np
X = np.load('ml_results/feature_matrix.npy')
print('Shape X:', X.shape)   # atteso: (11000, 105)

import pandas as pd
names = pd.read_csv('ml_results/feature_names.csv')
print('Feature names:', names.head(10))
# atteso: C3_delta, C3_theta, C3_alpha, C3_beta, C3_gamma, C4_delta, ...
```

### Caricare un modello salvato per inference

```python
import joblib
import numpy as np

# Carica il modello
model = joblib.load('ml_results/saved_models/svm_rbf.joblib')

# Esempio: predici su una singola epoca (shape: 1, 105)
# (la feature extraction va eseguita prima con extract_psd_features)
from src.eeg_ml_pipeline import extract_psd_features
epoch = np.load('output/epochs/s-01_epochs.npy')[0]   # prima epoca
features = extract_psd_features(epoch).reshape(1, -1)
pred = model.predict(features)
prob = model.predict_proba(features)[0, 1]
print(f'Classe predetta: {pred[0]}  (P(vantaggioso)={prob:.3f})')
```

---

## 9. Riferimenti

- Chávez-Sánchez et al. (2026). *Scientific Data*, 13:359. https://doi.org/10.1038/s41597-026-06662-0
- Bechara et al. (1994). *Cognition*, 50, 7–15.
- Bianchin & Angrilli (2011). *Brain and Cognition*, 75, 273–280.
- Cui et al. (2013). *Frontiers in Human Neuroscience*, 7, 776.
- Balconi & Angioletti (2022). *Clinical EEG and Neuroscience*, 53, 268–277.
- Aram et al. (2019). *Sage Open*, 9(3).
- Pedregosa et al. (2011). *Journal of Machine Learning Research*, 12, 2825–2830. (scikit-learn)
- Gramfort et al. (2013). *Frontiers in Neuroscience*, 7, 267. (MNE-Python)
