from transformers import pipeline
import scipy
from transformers import VitsModel, AutoTokenizer
import torch
import os
import random
import scipy.io.wavfile as wavfile

emotion = "happy"
model_id = f"Beka-pika/mms_kaz_tts_{emotion}"  # Replace with your specific model ID
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

# Helper function to synthesize audio
def synthesize_and_save(text, speaker, emotion, output_file):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, speaker_id=speaker_id_dict[speaker]).waveform
        synthesized_audio = outputs.cpu().numpy().flatten()
        # Save audio file
        print(output_file)
        wavfile.write(output_file, rate = model.config.sampling_rate, data=synthesized_audio)

if __name__ == '__main__':
    speakers = ["399172782", "805570882", "1263201035"]

    dataset_path = "EmoKaz/"
    output_path = "generated/"

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    for speaker in speakers:
        speaker_path = os.path.join(dataset_path, speaker)
        if not os.path.isdir(speaker_path):
            continue  # Skip if not a directory

        text_files = []
        subset = "eval"
        subset_path = os.path.join(speaker_path, subset)

        if not os.path.exists(subset_path):
            continue

        for file in os.listdir(subset_path):
            if file.endswith(".txt") and file.split("_")[1] == emotion:
                text_files.append((speaker, subset, file))

        # Randomly select 10 samples
        random_samples = random.sample(text_files, min(10, len(text_files)))
        # print(random_samples)

        # Generate audio for selected samples
        for speaker, subset, file in random_samples:
            text_file = os.path.join(dataset_path, speaker, subset, file)
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Create output directories for speaker and emotion
            speaker_output_path = os.path.join(output_path, speaker, emotion)
            os.makedirs(speaker_output_path, exist_ok=True)

            # Generate audio and save
            audio_filename = os.path.splitext(file)[0] + ".wav"
            output_file = os.path.join(speaker_output_path, audio_filename)
            print(f"Generating audio for {speaker}, emotion: {emotion}, file: {audio_filename}")
            synthesize_and_save(text, speaker, emotion, output_file)