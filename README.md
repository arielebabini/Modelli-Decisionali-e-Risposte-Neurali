# Modelli Decisionali e Risposte Neurali

> **Classificazione EEG di decisioni vantaggiose e svantaggiose nell'Iowa Gambling Task (IGT)**  
> Corso di Interfaccia Uomo-Macchina · A.A. 2025-2026 · 3° anno magistrale

---

## 📋 Indice

- [Panoramica](#panoramica)
- [Dataset](#dataset)
- [Architettura del Progetto](#architettura-del-progetto)
- [Pipeline](#pipeline)
  - [Flusso completo](#flusso-completo)
  - [1. Preprocessing EEG](#1-preprocessing-eeg)
  - [2. Feature Extraction & ML](#2-feature-extraction--ml)
  - [3. Dataset Sintetico](#3-dataset-sintetico)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
  - [Comando unificato — run_pipeline.py](#comando-unificato--run_pipelinepy)
  - [Uso avanzato dei singoli script](#uso-avanzato-dei-singoli-script)
- [Struttura Output](#struttura-output)
- [Risultati](#risultati)
- [Documentazione](#documentazione)
- [Dipendenze](#dipendenze)

---

## Panoramica

Questo progetto implementa una pipeline end-to-end per lo studio delle **risposte neurali** associate alle scelte nell'**Iowa Gambling Task (IGT)** — un paradigma neuropsicologico classico per lo studio del processo decisionale sotto incertezza.

L'obiettivo è classificare le epoche EEG pre-decisionali (finestra **[-2s, 0s]** prima della scelta) in:

| Classe | Deck | Descrizione |
|--------|------|-------------|
| **0 — Disadvantageous** | A, B | Scelte rischiose, perdite nette a lungo termine |
| **1 — Advantageous**    | C, D | Scelte sicure, guadagni netti a lungo termine |

La classificazione è condotta su feature **PSD (Power Spectral Density)** estratte con il metodo di **Welch** sulle 5 bande neurofisiologiche principali (δ, θ, α, β, γ), usando tre classificatori in cross-validazione **LOSO (Leave-One-Subject-Out)**.

---

## Dataset

**Chávez-Sánchez et al. — Iowa Gambling Task EEG Dataset**  
_Scientific Data, Elsevier (2026)_  
🔗 [Mendeley Data — DOI: 10.17632/2pw2m39yct/2](https://data.mendeley.com/datasets/2pw2m39yct/2)

### Caratteristiche principali

| Parametro | Valore |
|-----------|--------|
| Soggetti | ~20 partecipanti (gruppi ENG / PCS) |
| Canali EEG | 21 elettrodi (sistema internazionale 10-20) |
| Frequenza di campionamento | 256 Hz |
| Durata sessione | ~16 minuti |
| Trial IGT | 200 decisioni per soggetto |
| Riferimento acquisizione | Pz |

### Canali EEG (21 elettrodi)

```
C3, C4, O1, O2, A1, A2, Cz,
F3, F4, F7, F8, Fz,
Fp1, Fp2, Fpz, P3, P4,
T4, T5, T6, Pz
```

> ⚠️ Il dataset reale deve essere scaricato manualmente da Mendeley e posizionato in `eeg_igt_pipeline/data/igt_eeg_dataset/`.  
> È disponibile un **generatore di dati sintetici** per testare la pipeline senza il dataset reale (vedi [Dataset Sintetico](#3-dataset-sintetico)).

---

## Architettura del Progetto

```
Modelli-Decisionali-e-Risposte-Neurali/
│
├── README.md                          ← questo file
│
├── Documentazione/
│   ├── documentazione_tecnica_eeg.tex ← sorgente LaTeX
│   └── documentazione_tecnica_eeg.pdf ← documentazione tecnica completa
│
└── eeg_igt_pipeline/
    ├── requirements.txt               ← dipendenze Python
    ├── run_pipeline.py                ← ★ orchestratore unificato (entry-point)
    │
    ├── src/
    │   ├── eeg_preprocessing.py       ← modulo: preprocessing EEG con MNE
    │   ├── eeg_ml_pipeline.py         ← modulo: feature extraction + ML
    │   └── generate_synthetic_dataset.py ← modulo: generatore dati sintetici
    │
    ├── data/
    │   ├── igt_eeg_dataset/           ← dataset reale Mendeley (da scaricare)
    │   └── synthetic/                 ← dataset sintetico (generato)
    │
    └── pipeline_output/               ← output unificato (run_pipeline.py)
        ├── data_synthetic/            ← dataset sintetico generato
        ├── preprocessing/             ← output preprocessing
        │   ├── epochs/                ← epoche .npy per soggetto
        │   │   ├── s-01_epochs.npy
        │   │   ├── s-01_labels.npy
        │   │   ├── s-01_samples.npy
        │   │   └── s-01_info.npz
        │   ├── figures/               ← figure diagnostiche per soggetto
        │   └── pipeline_summary.csv
        └── ml_results/                ← output pipeline ML
            ├── results_summary.csv
            ├── confusion_matrices.png
            ├── roc_curves.png
            ├── feature_importance.png
            ├── metrics_boxplot.png
            └── saved_models/
```

---

## Pipeline

### Flusso completo

I tre moduli Python sono progettati per funzionare sia **indipendentemente** (uso avanzato) sia tramite l'**orchestratore unificato** [`run_pipeline.py`](eeg_igt_pipeline/run_pipeline.py):

```
┌─────────────────────────────────────────────────────────────────┐
│                     run_pipeline.py                             │
│                  (orchestratore unificato)                      │
│                                                                 │
│  STEP 0              STEP 1                 STEP 2             │
│  ─────────────       ──────────────────     ──────────────────  │
│  generate_           eeg_preprocessing      eeg_ml_pipeline    │
│  synthetic_          .py                    .py                │
│  dataset.py          │                      │                  │
│  │                   │  epochs .npy         │  results CSV     │
│  │  dataset CSV      │  labels .npy     ──► │  figure PNG      │
│  └─────────────────► └──────────────────    │  modelli .joblib │
│   (solo --mode        (preprocessing MNE,   └──────────────────│
│    synthetic)          ICA, filtri, epoche)  (PSD+LOSO CV)     │
└─────────────────────────────────────────────────────────────────┘
```

---

### 1. Preprocessing EEG

**File:** [`src/eeg_preprocessing.py`](eeg_igt_pipeline/src/eeg_preprocessing.py)

La pipeline di preprocessing si articola in **10 step** per ogni soggetto:

```
Raw EEG (CSV, µV)
       │
       ▼
 1. Caricamento EEG + dati comportamentali IGT
       │
       ▼
 2. Creazione oggetto MNE RawArray (µV → V)
       │
       ▼
 3. Re-referencing: Common Average Reference (CAR)
       │
       ▼
 4. Filtraggio bandpass FIR: 0.5–70 Hz (Hamming)
    + Filtro notch: 50 Hz (standard europeo)
       │
       ▼
 5. ICA (FastICA, 15 componenti)
    → Rimozione automatica artefatti oculari (Fp1, Fp2, Fpz)
       │
       ▼
 6. Normalizzazione z-score per canale
       │
       ▼
 7. Sincronizzazione EEG–IGT (colonna EEG_SAMPLE)
       │
       ▼
 8. Estrazione epoche [-2s, 0s] pre-decisione
       │
       ▼
 9. Salvataggio .npy per soggetto
       │
       ▼
10. Figure diagnostiche (PSD, ERP, raw vs. clean)
```

#### Bande di frequenza analizzate

| Banda | Range (Hz) | Ruolo neurofisiologico |
|-------|-----------|------------------------|
| **Delta** (δ) | 0.5 – 4 | Processi profondi, sonno, attenzione sostenuta |
| **Theta** (θ) | 4 – 8 | Memoria di lavoro, corteccia prefrontale, deliberazione |
| **Alpha** (α) | 8 – 13 | Rilassamento, inibizione corticale, attenzione |
| **Beta** (β)  | 13 – 30 | Pensiero attivo, controllo motorio, concentrazione |
| **Gamma** (γ) | 30 – 45 | Integrazione cognitiva, binding, elaborazione rapida |

---

### 2. Feature Extraction & ML

**File:** [`src/eeg_ml_pipeline.py`](eeg_igt_pipeline/src/eeg_ml_pipeline.py)

```
Epoche .npy (n_epochs × 21 canali × 512 campioni)
       │
       ▼
 1. PSD Welch per canale → 5 bande → vettore feature
    Shape: (n_epochs, 21 canali × 5 bande) = (n_epochs, 105)
       │
       ▼
 2. Definizione modelli (sklearn Pipeline con StandardScaler)
    ├── Logistic Regression (L2, class_weight=balanced)
    ├── SVM RBF (C=1.0, gamma=scale, probability=True)
    └── Random Forest (200 alberi, min_samples_leaf=2)
       │
       ▼
 3. Cross-validazione LOSO (Leave-One-Subject-Out)
    → Parallelizzazione joblib su tutti i core
    → Nessun data leakage (scaler fittato solo su train)
       │
       ▼
 4. Metriche aggregate (media ± std per fold)
    Accuracy · Precision · Recall · F1 · ROC-AUC
       │
       ▼
 5. Visualizzazioni
    ├── Confusion Matrix (normalizzata, aggregata LOSO)
    ├── ROC Curve (media ± std per modello)
    ├── Feature Importance (RF Gini + LR coefficienti)
    └── Boxplot metriche per fold
       │
       ▼
 6. Salvataggio modelli finali (.joblib)
```

#### Garanzie anti-leakage

- `StandardScaler` è **dentro** la sklearn `Pipeline` → fit esclusivamente sul train set di ogni fold
- La strategia **LOSO** garantisce la separazione per soggetto (nessun epoch dello stesso soggetto in train e test)
- Ogni fold **clona** la pipeline → nessuna contaminazione tra processi paralleli

---

### 3. Dataset Sintetico

**File:** [`src/generate_synthetic_dataset.py`](eeg_igt_pipeline/src/generate_synthetic_dataset.py)

Genera dati EEG-IGT sintetici con la **stessa struttura** del dataset reale Mendeley, per permettere il testing dell'intera pipeline senza download del dataset.

Caratteristiche del segnale sintetico:
- **Pink noise (1/f)** come sfondo realistico
- **Componente alpha** (9–11 Hz) prominente su O1/O2 (regioni occipitali)
- **Componente theta** (6 Hz) prominente su Fz/F3/F4 (frontale)
- **Artefatti blink** simulati su Fp1, Fp2, Fpz ogni ~5s
- **Curva di apprendimento IGT**: probabilità di scelta advantageous aumenta da 30% a 65% nel corso dei 200 trial

---

## Installazione

### Prerequisiti

- Python **3.10+**
- macOS / Linux / Windows (WSL consigliato su Windows)

### Setup ambiente virtuale

```bash
cd eeg_igt_pipeline

# Crea e attiva l'ambiente virtuale
python -m venv venv
source venv/bin/activate      # macOS/Linux
# .\venv\Scripts\activate    # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

### Dipendenze principali

```
mne>=1.6.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## Utilizzo

### Comando unificato — run_pipeline.py

[`run_pipeline.py`](eeg_igt_pipeline/run_pipeline.py) è l'**entry-point principale** del progetto: esegue in sequenza la generazione/caricamento dati, il preprocessing e la classificazione ML con un singolo comando.

```bash
cd eeg_igt_pipeline
source venv/bin/activate
```

#### ▶ Test rapido con dati sintetici (zero download)

```bash
# Genera 10 soggetti sintetici, preprocessa e classifica
python run_pipeline.py --mode synthetic
```

#### ▶ Più soggetti, senza figure diagnostiche

```bash
python run_pipeline.py --mode synthetic --n_subjects 20 --no-figures
```

#### ▶ Dataset reale Mendeley — pipeline completa LOSO

```bash
python run_pipeline.py \
    --mode real \
    --dataset ./data/igt_eeg_dataset
```

#### ▶ Dataset reale — debug (primi 5 soggetti, GroupKFold)

```bash
python run_pipeline.py \
    --mode real \
    --dataset ./data/igt_eeg_dataset \
    --limit 5 \
    --cv group_kfold --n_splits 5 \
    --output-root ./debug_output
```

#### Opzioni CLI — `run_pipeline.py`

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--mode` | `synthetic` | `synthetic` (genera dati) o `real` (dataset Mendeley) |
| `--dataset` | — | Directory root dataset reale _(richiesto con `--mode real`)_ |
| `--output-root` | `./pipeline_output` | Directory radice per tutti gli output |
| `--n_subjects` | `10` | Soggetti sintetici da generare _(solo `--mode synthetic`)_ |
| `--seed` | `42` | Seed random per la generazione sintetica |
| `--no-figures` | `False` | Disabilita figure diagnostiche nel preprocessing |
| `--limit` | `None` | Elabora solo i primi N soggetti |
| `--cv` | `loso` | Strategia CV: `loso` o `group_kfold` |
| `--n_splits` | `5` | k per GroupKFold |
| `--n_jobs` | `-1` | Processi paralleli per la CV (-1 = tutti i core) |
| `--no-save-models` | `False` | Non salvare i modelli `.joblib` finali |

---

### Uso avanzato dei singoli script

I tre moduli possono essere eseguiti separatamente per avere controllo granulare su ogni fase.

#### Generazione dataset sintetico

```bash
python src/generate_synthetic_dataset.py \
    --output ./data/synthetic \
    --n_subjects 5 \
    --seed 42
```

#### Preprocessing EEG

```bash
python src/eeg_preprocessing.py \
    --dataset ./data/igt_eeg_dataset \
    --output ./output \
    --limit 5          # opzionale: solo i primi N soggetti
```

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--dataset` | _(richiesto)_ | Directory root del dataset |
| `--output` | `./output` | Directory output |
| `--no-figures` | `False` | Disabilita figure diagnostiche |
| `--limit` | `None` | Elabora solo i primi N soggetti |

#### ML Pipeline

```bash
python src/eeg_ml_pipeline.py \
    --mode real \
    --input ./output \
    --output ./ml_results_loso \
    --cv loso \
    --n_jobs -1
```

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--mode` | `synthetic` | `synthetic` (dati interni) o `real` (legge da `--input`) |
| `--input` | `./output` | Directory output del preprocessing |
| `--output` | `./ml_results` | Directory output ML |
| `--cv` | `loso` | `loso` o `group_kfold` |
| `--n_splits` | `5` | k per GroupKFold |
| `--n_subjects` | `10` | Soggetti sintetici _(solo `--mode synthetic`)_ |
| `--n_jobs` | `-1` | Processi paralleli |
| `--no_save_models` | `False` | Non salvare i modelli `.joblib` finali |

---

## Struttura Output

Usando `run_pipeline.py` tutto l'output viene organizzato sotto `--output-root` (default: `./pipeline_output`).

```
pipeline_output/
│
├── data_synthetic/               ← dataset sintetico (solo --mode synthetic)
│   ├── participants.csv
│   ├── s-01/
│   │   ├── s-01_eeg.csv
│   │   └── s-01_igt.csv
│   └── ...
│
├── preprocessing/                ← output Step 1 (eeg_preprocessing.py)
│   ├── pipeline_summary.csv      ← statistiche per soggetto
│   ├── epochs/
│   │   ├── s-01_epochs.npy       ← shape: (n_epochs, 21, 512)
│   │   ├── s-01_labels.npy       ← shape: (n_epochs,) — 0=disadv / 1=adv
│   │   ├── s-01_samples.npy      ← indici campione EEG sincronizzati
│   │   └── s-01_info.npz         ← metadati (ch_names, sfreq, tmin, tmax)
│   └── figures/
│       ├── s-01_raw_vs_clean.png ← confronto EEG prima/dopo preprocessing
│       ├── s-01_psd_pre_filter.png
│       ├── s-01_psd_post_filter.png
│       └── s-01_erp_mean.png     ← ERP medio per classe (adv vs. disadv)
│
└── ml_results/                   ← output Step 2 (eeg_ml_pipeline.py)
    ├── results_summary.csv       ← media ± std per modello e metrica
    ├── feature_matrix.npy        ← matrice feature (n_epochs, 105)
    ├── feature_names.csv         ← nomi delle 105 feature (canale_banda)
    ├── feature_ranking.csv       ← ranking importanza feature (RF + LR)
    ├── confusion_matrices.png    ← CM normalizzata aggregata LOSO
    ├── roc_curves.png            ← ROC media ± std per modello
    ├── feature_importance.png    ← top-20 feature per RF e LR
    ├── metrics_boxplot.png       ← boxplot metriche per fold
    └── saved_models/
        ├── logistic_regression.joblib
        ├── svm_rbf.joblib
        └── random_forest.joblib
```

---

## Risultati

Risultati ottenuti su dati sintetici (LOSO, 10 soggetti, seed=42):

| Modello | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.521 ± 0.098 | 0.529 ± 0.179 | 0.520 ± 0.277 | 0.475 ± 0.188 | 0.545 ± 0.063 |
| **SVM (RBF)**          | 0.520 ± 0.099 | 0.534 ± 0.165 | 0.451 ± 0.289 | 0.431 ± 0.199 | 0.553 ± 0.068 |
| **Random Forest**      | 0.513 ± 0.084 | 0.529 ± 0.133 | 0.546 ± 0.259 | 0.492 ± 0.159 | 0.532 ± 0.059 |

> **Nota:** I risultati su dati sintetici sono vicini alla baseline casuale (AUC ≈ 0.50), come atteso — il segnale differenziale tra classi nei dati sintetici è intenzionalmente debole. I risultati su dati reali sono riportati nella documentazione tecnica.

---

## Documentazione

La documentazione tecnica completa è disponibile in:

- **PDF:** [`Documentazione/documentazione_tecnica_eeg.pdf`](Documentazione/documentazione_tecnica_eeg.pdf)
- **Sorgente LaTeX:** [`Documentazione/documentazione_tecnica_eeg.tex`](Documentazione/documentazione_tecnica_eeg.tex)

La documentazione include:
- Fondamenti neuroscientifici del IGT e della risposta EEG
- Dettaglio di ogni step della pipeline con motivazioni metodologiche
- Analisi delle bande di frequenza di interesse
- Validazione della pipeline e analisi degli artefatti ICA
- Discussione dei risultati e limitazioni

---

## Dipendenze

| Libreria | Versione minima | Utilizzo |
|----------|-----------------|---------|
| `mne` | ≥ 1.6.0 | Preprocessing EEG, ICA, gestione epoche |
| `numpy` | ≥ 1.24.0 | Array operations, salvataggio .npy |
| `pandas` | ≥ 2.0.0 | Caricamento CSV, riepilogo risultati |
| `scipy` | ≥ 1.11.0 | PSD Welch, filtraggio |
| `matplotlib` | ≥ 3.7.0 | Visualizzazioni (backend Agg) |
| `seaborn` | ≥ 0.12.0 | Confusion matrix heatmap |
| `scikit-learn` | ≥ 1.3.0 | Pipeline ML, CV, metriche, classificatori |

---

## Autori

Progetto sviluppato nell'ambito del corso di **Interfaccia Uomo-Macchina**  
Anno accademico 2025-2026 · Università degli Studi

Babini Ariele - 757608
Bottaro Federico - 758017
Nicholas Maria - 757386

---

## Licenza

Questo progetto è sviluppato a scopo didattico.  
Il dataset originale è soggetto alla licenza definita dai rispettivi autori su [Mendeley Data](https://data.mendeley.com/datasets/2pw2m39yct/2).
