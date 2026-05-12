"""
=============================================================================
Generatore dataset sintetico – IGT EEG
=============================================================================
Crea dati sintetici con la stessa struttura del dataset reale Mendeley
per permettere di testare la pipeline SENZA scaricare il dataset.

Uso:
    python generate_synthetic_dataset.py --output ./data/synthetic --n_subjects 5

Struttura generata:
    ./data/synthetic/
    ├── participants.csv
    ├── s-01/
    │   ├── s-01_eeg.csv      (EEG raw, µV)
    │   └── s-01_igt.csv      (behavioral data)
    ├── s-02/
    │   └── ...
=============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Parametri fissi che rispecchiano il dataset reale
SFREQ       = 256      # Hz
N_CHANNELS  = 21
DURATION_S  = 960      # ~16 min come da paper
N_TRIALS    = 200      # 200 decisioni per soggetto
DECKS       = ["A", "B", "C", "D"]

CH_NAMES = [
    "C3", "C4", "O1", "O2", "A1", "A2", "Cz",
    "F3", "F4", "F7", "F8", "Fz",
    "Fp1", "Fp2", "Fpz", "P3", "P4",
    "T4", "T5", "T6", "Pz"
]


def simulate_eeg_signal(
    n_samples: int,
    n_channels: int,
    sfreq: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Genera un segnale EEG sintetico realistico (µV).

    Modella:
      - componente α (8–12 Hz) prominente nelle regioni posteriori
      - componente θ (4–8 Hz) prominente frontalmente (Fz, F3, F4)
      - rumore 1/f (pink noise) come sfondo
      - artefatto blink simulato su Fp1, Fp2, Fpz ogni ~5s
    """
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_samples, n_channels))

    for ch_idx in range(n_channels):
        # Pink noise (1/f): somma di sinusoidi con ampiezza ∝ 1/f
        pink = np.zeros(n_samples)
        for freq in np.arange(1, 60, 0.5):
            amp = 5.0 / freq
            phase = rng.uniform(0, 2 * np.pi)
            pink += amp * np.sin(2 * np.pi * freq * t + phase)

        # Componente alpha (occipitale) - canali O1, O2 (indici 2,3)
        if ch_idx in [2, 3]:
            alpha_amp = rng.uniform(20, 40)
            alpha_freq = rng.uniform(9, 11)
            pink += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

        # Componente theta (frontale) - Fz, F3, F4 (indici 11, 7, 8)
        if ch_idx in [7, 8, 11]:
            theta_amp = rng.uniform(10, 20)
            pink += theta_amp * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2*np.pi))

        # Rumore bianco di fondo
        pink += rng.normal(0, 2.0, n_samples)

        data[:, ch_idx] = pink

    # Artefatti blink su canali frontali (Fp1=12, Fp2=13, Fpz=14)
    blink_channels = [12, 13, 14]
    blink_interval = int(5 * sfreq)  # ogni 5 secondi
    blink_duration = int(0.3 * sfreq)  # 300 ms
    for t_start in range(0, n_samples - blink_duration, blink_interval):
        blink_shape = np.exp(-0.5 * ((np.arange(blink_duration) - blink_duration/2) /
                                      (blink_duration/6))**2) * 150
        for ch in blink_channels:
            data[t_start:t_start + blink_duration, ch] += blink_shape

    return data.astype(np.float32)


def simulate_igt_behavior(
    n_trials: int,
    n_eeg_samples: int,
    sfreq: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Genera dati comportamentali IGT sintetici.

    Simula un apprendimento progressivo: il partecipante tende a selezionare
    deck svantaggiosi (A/B) nei primi trial e quelli vantaggiosi (C/D)
    nei trial successivi (curva di apprendimento classica del IGT).
    """
    # Probabilità di scegliere C/D aumenta nel tempo (apprendimento)
    prob_advantageous = np.linspace(0.3, 0.65, n_trials)

    decks_chosen = []
    for i in range(n_trials):
        if rng.random() < prob_advantageous[i]:
            deck = rng.choice(["C", "D"])
        else:
            deck = rng.choice(["A", "B"])
        decks_chosen.append(deck)

    # Struttura temporale: baseline 3min, poi trial ogni ~3–5s
    baseline_samples = int(3 * 60 * sfreq)
    trial_samples = []
    current_sample = baseline_samples

    for i in range(n_trials):
        trial_samples.append(int(current_sample))
        # Intervallo inter-trial: RT (1–3s) + feedback (1.5s)
        iti = rng.uniform(2.5, 4.5)
        current_sample += int(iti * sfreq)
        # Pausa 4 min dopo il trial 100
        if i == 99:
            current_sample += int(4 * 60 * sfreq)

    # Clip ai campioni disponibili
    trial_samples = [min(s, n_eeg_samples - 1) for s in trial_samples]

    # Win/Loss per deck (secondo distribuzione originale IGT)
    win_loss = {
        "A": (100,  lambda r: -r.choice([150, 200, 250, 300, 350],
                                         p=[0.5, 0.1, 0.1, 0.1, 0.2])),
        "B": (100,  lambda r: -r.choice([0, 0, 0, 0, 1250],
                                         p=[0.9, 0.025, 0.025, 0.025, 0.025])),
        "C": (50,   lambda r: -r.choice([25, 50, 75],
                                         p=[0.5, 0.25, 0.25])),
        "D": (50,   lambda r: -r.choice([0, 0, 0, 0, 250],
                                         p=[0.9, 0.025, 0.025, 0.025, 0.025])),
    }

    rows = []
    total_credits = 2000
    for i, (deck, eeg_sample) in enumerate(zip(decks_chosen, trial_samples)):
        win_val, loss_fn = win_loss[deck]
        win = win_val
        loss = loss_fn(rng)
        net = win + loss
        total_credits += net
        rt = int(rng.uniform(800, 3000))  # ms

        rows.append({
            "TRIAL":       i + 1,
            "DECK":        deck,
            "WIN":         win,
            "LOSS":        loss,
            "NET":         net,
            "TOTAL":       total_credits,
            "RT":          rt,
            "EEG_SAMPLE":  eeg_sample,
        })

    return pd.DataFrame(rows)


def generate_dataset(output_root: str, n_subjects: int = 5, seed: int = 42) -> None:
    """
    Genera il dataset sintetico completo.

    Parameters
    ----------
    output_root : str  – directory di output
    n_subjects  : int  – numero di soggetti da generare
    seed        : int  – seed random base
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    n_eeg_samples = DURATION_S * SFREQ
    participants_rows = []

    print(f"Generazione dataset sintetico: {n_subjects} soggetti")
    print(f"  EEG: {N_CHANNELS} canali, {SFREQ} Hz, {DURATION_S}s "
          f"({n_eeg_samples} campioni)")
    print(f"  IGT: {N_TRIALS} trial per soggetto\n")

    for i in range(1, n_subjects + 1):
        sid = f"s-{i:02d}"
        rng = np.random.default_rng(seed + i)
        subj_dir = root / sid
        subj_dir.mkdir(exist_ok=True)

        # --- Genera EEG ---
        eeg = simulate_eeg_signal(n_eeg_samples, N_CHANNELS, SFREQ, rng)
        eeg_df = pd.DataFrame(eeg, columns=CH_NAMES)
        eeg_df.to_csv(subj_dir / f"{sid}_eeg.csv", index=False)

        # --- Genera IGT ---
        igt_df = simulate_igt_behavior(N_TRIALS, n_eeg_samples, SFREQ, rng)
        igt_df.to_csv(subj_dir / f"{sid}_igt.csv", index=False)

        # --- Partecipanti ---
        sex = "M" if rng.random() > 0.5 else "F"
        group = rng.choice(["ENG", "PCS"])
        age = int(rng.integers(20, 25))
        participants_rows.append({
            "participant_id": sid,
            "sex": sex,
            "age": age,
            "group": group
        })

        deck_dist = igt_df["DECK"].value_counts().to_dict()
        adv = deck_dist.get("C", 0) + deck_dist.get("D", 0)
        print(f"  {sid}: {sex}, {age}y, {group} | "
              f"EEG {eeg.shape} | "
              f"IGT {len(igt_df)} trial | "
              f"Adv={adv}, Disadv={N_TRIALS-adv}")

    # --- participants.csv ---
    pd.DataFrame(participants_rows).to_csv(root / "participants.csv", index=False)
    print(f"\nDataset sintetico generato in: {root.resolve()}")
    print("Pronto per testare la pipeline!")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset EEG-IGT sintetico per testing"
    )
    parser.add_argument("--output", type=str, default="./data/synthetic",
                        help="Directory output (default: ./data/synthetic)")
    parser.add_argument("--n_subjects", type=int, default=5,
                        help="Numero di soggetti da generare (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed random (default: 42)")
    args = parser.parse_args()

    generate_dataset(
        output_root=args.output,
        n_subjects=args.n_subjects,
        seed=args.seed
    )
