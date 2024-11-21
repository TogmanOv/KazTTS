import kaldiio
import os
import librosa
from tqdm import tqdm
import glob
import json 
from shutil import copyfile
import pandas as pd
import argparse
# from text import _clean_text, symbols
from num2words import num2words
import re
# from melspec import mel_spectrogram
import torchaudio
import numpy as np
from pathlib import Path

dataset_path = "/home/beknur.kalmakhanbet/Documents/AI701/Project/EmoKaz"

data = []

def extract_prosodic_features(audio_path):
    y, sr = librosa.load(audio_path)

    pitches, magnitude = librosa.piptrack(y=y, sr=sr)
    pitch = np.max(pitches, axis=0)
    pitch = pitch[pitch > 0] # filter out zero values
    avg_pitch = np.mean(pitch) if len(pitch) > 0 else 0

    # Energy
    energy = librosa.feature.rms(y=y).flatten()
    avg_energy = np.mean(energy)

    duration = librosa.get_duration(y=y, sr=sr)

    return avg_pitch, avg_energy, duration

if __name__ == '__main__':
    for speaker in Path(dataset_path).iterdir():
        if speaker.is_dir():
            for set_type in ["train", "eval"]:
                set_path = speaker / set_type
                for wav_file in set_path.glob("*.wav"):
                    print(wav_file)
                    base_name = wav_file.stem
                    text_file = set_path / f"{base_name}.txt"

                    with open(text_file, "r") as f:
                        transcript = f.readline().strip()
                        emotion = base_name.split("_")[1]
                    
                    avg_pitch, avg_energy, duration = extract_prosodic_features(str(wav_file))

                    data.append({
                        "speaker": speaker.name,
                        "set_type": set_type,
                        "file": wav_file.name,
                        "transcript": transcript,
                        "emotion": emotion,
                        "avg_pitch": avg_pitch,
                        "avg_energy": avg_energy,
                        "duration": duration
                    })

    df = pd.DataFrame(data)
    df.to_csv("base_data_preprocess.csv")