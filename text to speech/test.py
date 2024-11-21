from transformers import VitsModel, AutoTokenizer
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-kaz")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kaz")

text = "Ақынның ғұмырлы поэзиясы бүгінгі таңда барша жұрттың жүрегінен берік орын алып, қуанышы мен қайғысын бөлісетін айнымас жан серігіне айналды."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

import scipy

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output.squeeze().cpu().numpy())