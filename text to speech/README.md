# Text to Speech

## Overview

This module focuses on building an emotional tts model for the Kazakh language. The model is fine-tuned to the Facebook/mms-tts-kaz model to synthesize audio based on text. The dataset used for fine-tuning is the EmoKaz, the new dataset (2024) focusing on six emotions: anger, happiness, fear, neutral, surprise, and sad. The finetuning script was adopted from [ylacombe](https://github.com/ylacombe/finetune-hf-vits). 

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requireemnts](#requirements)

---

## Repository Structure

```plaintext
finetune-hf-vits/       # Folder for fine-tuning VITS models
README.md               # Documentation for the repository
calculate_mcd.py        # Script to compute Mean Cepstral Distortion (MCD)
dataset.py              # Dataset handler for managing and loading data for testing dataset
normalize.py            # Script to normalize dataset features
prepare_mos.py          # Script for preparing MOS evaluation files
preprocess.py           # Preprocessing script for dataset preparation
test_finetune.py        # Script to test the fine-tuned VITS model
```
---

## Requirements
