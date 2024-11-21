from transformers import pipeline
import scipy
from transformers import VitsModel, AutoTokenizer
import torch

model_id = "Beka-pika/mms_kaz_tts_angry"  # Replace with your specific model ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VitsModel.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# angry
speaker_id_dict = {
    '1263201035': 1, '399172782': 0, '805570882': 2
}

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
# speaker_id_dict = {
#     '1263201035': 0, '399172782': 2, '805570882': 1
# }

text = "Ақынның ғұмырлы поэзиясы бүгінгі таңда барша жұрттың жүрегінен берік орын алып, қуанышы мен қайғысын бөлісетін айнымас жан серігіне айналды."
speaker = '399172782'

inputs = tokenizer(text, return_tensors="pt").to(device)
    
# Synthesize audio using the TTS model
with torch.no_grad():
    # Adjust the speaker_id parameter as per your model's requirements
    outputs = model(**inputs, speaker_id=speaker_id_dict[speaker]).waveform
    synthesized_audio = outputs.cpu().numpy().flatten()

scipy.io.wavfile.write("finetuned_output.wav", rate = model.config.sampling_rate, data=synthesized_audio)