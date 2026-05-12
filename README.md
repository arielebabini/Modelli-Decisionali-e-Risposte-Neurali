# EEG-Based Decision Making Classification using Iowa Gambling Task

## Project Overview

This project was developed for the course:

**Interfacce Uomo-Macchina – A.A. 2025/2026**  
Università degli Studi dell'Insubria

The objective of the project is to develop a complete EEG signal processing and machine learning pipeline capable of classifying decision-making states during the Iowa Gambling Task (IGT).

The project is based on the scientific dataset described in:

> Chávez-Sánchez M. et al.  
> *Behavioral and electroencephalographic dataset simultaneously acquired during the Iowa Gambling Task*  
> Scientific Data, 2026.

---

# Objective

The goal is to classify:

- **Advantageous decisions**
vs
- **Disadvantageous decisions**

using EEG activity recorded before the subject’s decision during the Iowa Gambling Task.

The classification is performed using EEG windows extracted before each decision event.

---

# Dataset

Dataset source:

- Iowa Gambling Task EEG Dataset
- 59 healthy participants
- 21 EEG channels
- Sampling rate: 256 Hz
- Behavioral + EEG synchronized data

Dataset includes:

- Raw EEG recordings
- Preprocessed EEG
- Behavioral IGT data
- Decision timestamps (`EEG SAMPLE`)
- Demographic information

---

# Project Pipeline

The implemented workflow includes:

## 1. EEG Preprocessing

- EEG loading
- Re-referencing
- Band-pass filtering (0.5–70 Hz)
- Notch filtering
- Artifact removal using ICA

---

## 2. EEG Synchronization

Behavioral IGT events are synchronized with EEG signals using the provided:

```text
EEG SAMPLE