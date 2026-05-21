"""
=============================================================================
run_pipeline.py – Orchestratore unificato EEG-IGT
=============================================================================
Esegue in sequenza:
  1. (Opzionale) Generazione dataset sintetico
  2. Preprocessing EEG  (eeg_preprocessing.py)
  3. Feature Extraction + ML  (eeg_ml_pipeline.py)

Uso rapido – modalità sintetica (nessun dataset da scaricare):
    python run_pipeline.py --mode synthetic

Uso completo – dataset reale Mendeley:
    python run_pipeline.py --mode real --dataset ./data/igt_eeg_dataset

Tutte le opzioni avanzate:
    python run_pipeline.py --help

=============================================================================
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging condiviso
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Utility: stampa sezione
# ---------------------------------------------------------------------------
def _section(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


# ---------------------------------------------------------------------------
# STEP 0 – Generazione dataset sintetico
# ---------------------------------------------------------------------------
def step_generate_synthetic(
    output: str,
    n_subjects: int,
    seed: int,
) -> str:
    """
    Genera il dataset sintetico e restituisce il percorso della directory creata.

    Parameters
    ----------
    output     : directory di output per il dataset sintetico
    n_subjects : numero di soggetti da generare
    seed       : seed random

    Returns
    -------
    str – percorso assoluto della directory del dataset generato
    """
    from src.generate_synthetic_dataset import generate_dataset  # import locale

    _section(f"STEP 0 – Generazione dataset sintetico ({n_subjects} soggetti)")
    t0 = time.time()

    generate_dataset(
        output_root=output,
        n_subjects=n_subjects,
        seed=seed,
    )

    elapsed = time.time() - t0
    logger.info(f"Dataset sintetico pronto in {elapsed:.1f}s → {Path(output).resolve()}")
    return output


# ---------------------------------------------------------------------------
# STEP 1 – Preprocessing EEG
# ---------------------------------------------------------------------------
def step_preprocessing(
    dataset_root: str,
    output_dir: str,
    save_figures: bool,
    subject_limit: int | None,
) -> str:
    """
    Esegue il preprocessing EEG su tutti i soggetti del dataset.

    Chiama direttamente run_full_pipeline() da eeg_preprocessing.py
    senza avviare un sottoprocesso, condividendo il processo Python
    e quindi il logging unificato.

    Parameters
    ----------
    dataset_root  : directory radice del dataset (reale o sintetico)
    output_dir    : directory dove salvare le epoche .npy
    save_figures  : se True, genera figure diagnostiche per soggetto
    subject_limit : se non None, elabora solo i primi N soggetti

    Returns
    -------
    str – percorso della directory di output del preprocessing
    """
    from src.eeg_preprocessing import run_full_pipeline  # import locale

    _section("STEP 1 – Preprocessing EEG")
    t0 = time.time()

    summary_df = run_full_pipeline(
        dataset_root=dataset_root,
        output_dir=output_dir,
        save_figures=save_figures,
        subject_limit=subject_limit,
    )

    n_ok = len(summary_df)
    elapsed = time.time() - t0
    logger.info(
        f"Preprocessing completato: {n_ok} soggetti in {elapsed:.1f}s "
        f"→ {Path(output_dir).resolve()}"
    )

    if n_ok == 0:
        raise RuntimeError(
            "Nessun soggetto elaborato correttamente. "
            "Controlla i file di input nella directory del dataset."
        )

    return output_dir


# ---------------------------------------------------------------------------
# STEP 2 – Feature Extraction + ML
# ---------------------------------------------------------------------------
def step_ml(
    input_dir: str,
    output_dir: str,
    cv_strategy: str,
    n_splits: int,
    n_jobs: int,
    save_models: bool,
) -> None:
    """
    Esegue la pipeline ML (PSD Welch + classificatori + LOSO CV).

    Chiama run_pipeline() da eeg_ml_pipeline.py importandolo come modulo,
    permettendo di riusare X e feature_names nel processo corrente
    senza duplicare la feature extraction.

    Parameters
    ----------
    input_dir    : directory output del preprocessing (contiene epochs/)
    output_dir   : directory di output per i risultati ML
    cv_strategy  : 'loso' | 'group_kfold'
    n_splits     : k per GroupKFold (ignorato in modalità loso)
    n_jobs       : processi paralleli per la CV (-1 = tutti i core)
    save_models  : se True, salva i modelli finali .joblib
    """
    from src.eeg_ml_pipeline import load_preprocessed_data, run_pipeline  # import locale

    _section("STEP 2 – Feature Extraction + Machine Learning")
    t0 = time.time()

    logger.info(f"Caricamento epoche da: {Path(input_dir).resolve()}")
    epochs_data, labels, subject_ids = load_preprocessed_data(input_dir)

    summary_df = run_pipeline(
        epochs_data=epochs_data,
        labels=labels,
        subject_ids=subject_ids,
        output_dir=output_dir,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        n_jobs=n_jobs,
        save_trained_models=save_models,
    )

    elapsed = time.time() - t0
    logger.info(f"ML pipeline completata in {elapsed:.1f}s → {Path(output_dir).resolve()}")

    # Riepilogo finale compatto
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    print("\n" + "=" * 65)
    print(f"  {'RISULTATI FINALI (media ± std) — ' + cv_strategy.upper()}")
    print("=" * 65)
    header = f"  {'Modello':<24}" + "".join(f"  {m[:7]:>9}" for m in metric_keys)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model in summary_df.index:
        row = f"  {model:<24}"
        for m in metric_keys:
            mean_v = summary_df.loc[model, f"{m}_mean"]
            std_v  = summary_df.loc[model, f"{m}_std"]
            row += f"  {mean_v:.3f}±{std_v:.3f}"
        print(row)
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Orchestratore EEG-IGT: genera/carica dati → preprocessing → ML.\n"
            "Combina generate_synthetic_dataset.py, eeg_preprocessing.py e "
            "eeg_ml_pipeline.py in un unico comando."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi d'uso:
  # Test rapido con dati sintetici (nessun dataset richiesto)
  python run_pipeline.py --mode synthetic

  # Dati sintetici, più soggetti, senza figure
  python run_pipeline.py --mode synthetic --n_subjects 20 --no-figures

  # Dataset reale Mendeley, LOSO completo
  python run_pipeline.py --mode real --dataset ./data/igt_eeg_dataset

  # Dataset reale, solo 5 soggetti (debug), GroupKFold
  python run_pipeline.py --mode real --dataset ./data/igt_eeg_dataset \\
      --limit 5 --cv group_kfold --n_splits 5
        """,
    )

    # --- Modalità ---
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help=(
            "'synthetic': genera dati sintetici e poi elabora (default). "
            "'real': usa il dataset reale in --dataset."
        ),
    )

    # --- Input ---
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Directory root del dataset reale (richiesto se --mode real). "
            "Struttura attesa: root/s-01/{s-01_eeg.csv, s-01_igt.csv}, ..."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output-root",
        type=str,
        default="./pipeline_output",
        help="Directory radice per tutti gli output (default: ./pipeline_output).",
    )

    # --- Dataset sintetico ---
    synth_group = parser.add_argument_group("Dati sintetici (--mode synthetic)")
    synth_group.add_argument(
        "--n_subjects",
        type=int,
        default=10,
        help="Numero di soggetti sintetici da generare (default: 10).",
    )
    synth_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed random per la generazione sintetica (default: 42).",
    )

    # --- Preprocessing ---
    prep_group = parser.add_argument_group("Preprocessing")
    prep_group.add_argument(
        "--no-figures",
        action="store_true",
        help="Disabilita la generazione di figure diagnostiche per soggetto.",
    )
    prep_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Elabora solo i primi N soggetti (utile per debug).",
    )

    # --- ML ---
    ml_group = parser.add_argument_group("Machine Learning")
    ml_group.add_argument(
        "--cv",
        choices=["loso", "group_kfold"],
        default="loso",
        help="Strategia di cross-validazione (default: loso).",
    )
    ml_group.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="k per GroupKFold (ignorato con --cv loso, default: 5).",
    )
    ml_group.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Processi paralleli per CV (-1 = tutti i core, default: -1).",
    )
    ml_group.add_argument(
        "--no-save-models",
        action="store_true",
        help="Non salvare i modelli finali (.joblib).",
    )

    args = parser.parse_args()

    # ---- Validazione argomenti ----
    if args.mode == "real" and args.dataset is None:
        parser.error("--dataset è richiesto con --mode real.")

    # ---- Directory di lavoro: sposta in eeg_igt_pipeline/ ----
    # Gli import locali (src.*) funzionano solo se la cwd è eeg_igt_pipeline/
    script_dir = Path(__file__).parent.resolve()
    import os
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    # ---- Percorsi output ----
    root_out      = Path(args.output_root).resolve()
    synthetic_dir = root_out / "data_synthetic"
    preproc_dir   = root_out / "preprocessing"
    ml_dir        = root_out / "ml_results"

    root_out.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    logger.info("=" * 60)
    logger.info("  ORCHESTRATORE EEG-IGT – AVVIO")
    logger.info(f"  Modalità : {args.mode.upper()}")
    logger.info(f"  Output   : {root_out}")
    logger.info("=" * 60)

    # ---- STEP 0: Generazione dataset (solo modalità synthetic) ----
    if args.mode == "synthetic":
        dataset_root = step_generate_synthetic(
            output=str(synthetic_dir),
            n_subjects=args.n_subjects,
            seed=args.seed,
        )
    else:
        dataset_root = args.dataset

    # ---- STEP 1: Preprocessing ----
    preproc_out = step_preprocessing(
        dataset_root=dataset_root,
        output_dir=str(preproc_dir),
        save_figures=not args.no_figures,
        subject_limit=args.limit,
    )

    # ---- STEP 2: Machine Learning ----
    step_ml(
        input_dir=preproc_out,
        output_dir=str(ml_dir),
        cv_strategy=args.cv,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        save_models=not args.no_save_models,
    )

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"  PIPELINE COMPLETA in {total_elapsed:.1f}s")
    logger.info(f"  Output root : {root_out}")
    logger.info(f"  Preprocessing: {preproc_dir}")
    logger.info(f"  ML results  : {ml_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
