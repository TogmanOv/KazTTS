import os
from transformers import pipeline
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from fastdtw import fastdtw
from pymcd.mcd import Calculate_MCD
from transformers import VitsModel, AutoTokenizer
import torch

# Load the TTS pipeline
model_id = "Beka-pika/mms_kaz_tts_happy"  # Replace with your specific model ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VitsModel.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# angry
# speaker_id_dict = {
#     '1263201035': 1, '399172782': 0, '805570882': 2
# }

# sad
# speaker_id_dict = {
#     '1263201035': 2, '399172782': 0, '805570882': 1
# }

# fear
# speaker_id_dict = {
#     '1263201035': 2, '399172782': 1, '805570882': 0
# }

# neutral
# speaker_id_dict = {
#     '1263201035': 0, '399172782': 2, '805570882': 1
# }

# surprise
# speaker_id_dict = {
#     '1263201035': 1, '399172782': 0, '805570882': 2
# }

# happy
speaker_id_dict = {
    '1263201035': 1, '399172782': 0, '805570882': 2
}

def compute_mfcc(audio, sr=16000, n_mfcc=13):
    """
    Extract MFCCs from audio data.
    
    Parameters:
    - audio (np.ndarray): Audio time series.
    - sr (int): Sampling rate.
    - n_mfcc (int): Number of MFCCs to extract.
    
    Returns:
    - mfccs (np.ndarray): MFCC feature matrix (frames x n_mfcc).
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Transpose to shape (frames, n_mfcc)

def calculate_mcd_with_dtw(mfcc1, mfcc2):
    """
    Calculate the Mel-Cepstral Distortion (MCD) between two MFCC sequences using DTW.
    
    Parameters:
    - mfcc1 (np.ndarray): MFCC sequence of the first audio (frames x coefficients).
    - mfcc2 (np.ndarray): MFCC sequence of the second audio (frames x coefficients).
    
    Returns:
    - mcd (float): Calculated MCD value in decibels (dB).
    """
    # Compute DTW alignment path
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
    
    # Accumulate distances along the optimal path
    total_distance = 0
    for i, j in path:
        diff = mfcc1[i] - mfcc2[j]
        total_distance += np.sqrt(np.sum(diff ** 2))
    
    # Apply MCD scaling factor
    mcd = (10.0 / np.log(10)) * (total_distance / len(path) + 1e-10)
    return mcd

# ==========================
# Synthesis and MCD Calculation
# ==========================

def synthesize_and_compute_mcd(text, ground_truth_path, speaker):
    """
    Synthesize speech from text and compute MCD with ground truth audio.
    
    Parameters:
    - text (str): Text to synthesize.
    - ground_truth_path (str): Path to the ground truth audio file.
    - speaker (str): Speaker identifier.
    
    Returns:
    - mcd (float): Calculated MCD value.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Synthesize audio using the TTS model
    with torch.no_grad():
        # Adjust the speaker_id parameter as per your model's requirements
        outputs = model(**inputs, speaker_id=speaker_id_dict[speaker]).waveform
        synthesized_audio = outputs.cpu().numpy().flatten()
    
    # Ensure mono audio
    if synthesized_audio.ndim > 1:
        synthesized_audio = np.mean(synthesized_audio, axis=1)
    
    # Get sampling rate from the model configuration
    synthesized_sr = model.config.sampling_rate
    
    # Load ground truth audio
    ground_truth_audio, ground_truth_sr = librosa.load(ground_truth_path, sr=None)

    if ground_truth_sr != synthesized_sr:
        ground_truth_audio = librosa.resample(
            y=ground_truth_audio, 
            orig_sr=ground_truth_sr, 
            target_sr=synthesized_sr
        )
    
    # Extract MFCCs
    mfcc_synthesized = compute_mfcc(synthesized_audio, sr=synthesized_sr)
    mfcc_ground_truth = compute_mfcc(ground_truth_audio, sr=synthesized_sr)
    
    # Calculate MCD
    mcd = calculate_mcd_with_dtw(mfcc_ground_truth, mfcc_synthesized)
    return mcd

# ==========================
# Dataset Processing
# ==========================

def calculate_mcd_for_dataset(dataset_path, target_emotion):
    """
    Calculate MCD for each speaker and overall in the dataset.
    
    Parameters:
    - dataset_path (str): Path to the dataset directory.
    - target_emotion (str): Target emotion label (e.g., 'happy').
    
    Returns:
    - speaker_mcds (dict): MCD values per speaker.
    - overall_avg_mcd (float): Overall average MCD across all speakers.
    """
    speakers = ["399172782", "805570882", "1263201035"]  # List of speaker IDs
    total_mcds = []
    speaker_mcds = {speaker: [] for speaker in speakers}
    
    # Iterate over each speaker
    for speaker in speakers:
        eval_path = os.path.join(dataset_path, speaker, "eval")
        if not os.path.isdir(eval_path):
            print(f"Warning: Evaluation directory not found for speaker {speaker}. Skipping...")
            continue
        
        print(f"\nCalculating MCD for speaker {speaker}...")
        
        # Process each .wav and .txt pair in the eval folder
        for file_name in tqdm(os.listdir(eval_path), desc=f"Speaker {speaker}"):
            if file_name.endswith(".wav") and file_name.split("_")[1] == target_emotion:
                audio_path = os.path.join(eval_path, file_name)
                text_path = audio_path.replace(".wav", ".txt")
                
                # Verify the existence of the corresponding text file
                if not os.path.isfile(text_path):
                    print(f"Warning: Text file not found for {file_name}. Skipping...")
                    continue
                
                # Read the text for synthesis
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Compute MCD between synthesized and ground truth audio
                try:
                    mcd = synthesize_and_compute_mcd(text, audio_path, speaker)
                    speaker_mcds[speaker].append(mcd)
                    total_mcds.append(mcd)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    continue
        
        # Calculate and print average MCD for this speaker
        if speaker_mcds[speaker]:
            avg_mcd_speaker = np.mean(speaker_mcds[speaker])
            print(f"Average MCD for speaker {speaker}: {avg_mcd_speaker:.2f} dB")
        else:
            print(f"No valid samples found for speaker {speaker}.")

    # Calculate and print overall average MCD across all speakers
    if total_mcds:
        overall_avg_mcd = np.mean(total_mcds)
        print(f"\nOverall Average MCD for all speakers: {overall_avg_mcd:.2f} dB")
    else:
        overall_avg_mcd = float('nan')
        print("\nNo MCD calculations were performed. Check your dataset and parameters.")
    
    return speaker_mcds, overall_avg_mcd

# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = "EmoKaz/"  # Replace with the actual path to your dataset
    
    # Define the target emotion (e.g., 'happy')
    target_emotion = "happy"
    
    # Calculate MCD for the dataset
    speaker_mcds, overall_avg_mcd = calculate_mcd_for_dataset(dataset_path, target_emotion)
