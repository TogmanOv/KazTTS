import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("base_data_preprocess.csv")

    pitch_mean = df["avg_pitch"].mean()
    pitch_std = df["avg_pitch"].std()

    energy_mean = df["avg_energy"].mean()
    energy_std = df["avg_energy"].std()

    duration_mean = df["duration"].mean()
    duration_std = df["duration"].std()

    df["normalized_pitch"] = (df["avg_pitch"] - pitch_mean) / pitch_std
    df["normalized_energy"] = (df["avg_energy"] - energy_mean) / energy_std

    df["normalized_duration"] = (df["duration"] - duration_mean) / duration_std

    df.to_csv("base_data_normalized.csv", index=False)