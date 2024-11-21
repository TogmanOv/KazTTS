# Text to Speech

This repository provides the scripts and configuration required to fine-tune [VITS (Variational Inference Text-to-Speech)] models using datasets with emotion labels. The repository supports data preprocessing, normalization, and evaluation tasks, facilitating a streamlined workflow for fine-tuning VITS on new datasets.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Preprocessing the Data](#preprocessing-the-data)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Preprocessing scripts to prepare datasets for fine-tuning.
- Support for emotion-based fine-tuning of the VITS model.
- Configurable training and evaluation pipeline.
- Scripts for calculating Mean Cepstral Distortion (MCD) and preparing MOS (Mean Opinion Score) evaluations.

---

## Repository Structure

```plaintext
finetune-hf-vits/       # Folder for fine-tuning VITS models
README.md               # Documentation for the repository
calculate_mcd.py        # Script to compute Mean Cepstral Distortion (MCD)
config.json             # Configuration file for the fine-tuning process
dataset.py              # Dataset handler for managing and loading data
normalize.py            # Script to normalize dataset features
prepare_mos.py          # Script for preparing MOS evaluation files
preprocess.py           # Preprocessing script for dataset preparation
test.py                 # Unit tests for fine-tuning scripts
test_finetune.py        # Script to test the fine-tuned VITS model
