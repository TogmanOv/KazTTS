import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import VitsModel, AutoTokenizer
import torch
from torch.optim import AdamW
from transformers import get_scheduler

# Load pre-trained model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-kaz")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kaz")

class EmotionalTTSDataset(Dataset):
    def __init__(self, df, label_encoder, max_text_len=200):
        self.df = df
        self.label_encoder = label_encoder
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]
        
        # Text as tensor (you may need to tokenize based on your model)
        text = row['transcript']
        text_tensor = torch.tensor(self.text_to_sequence(text), dtype=torch.long)

        # Audio features
        normalized_pitch = torch.tensor(row["normalized_pitch"], dtype=torch.float32)
        normalized_energy = torch.tensor(row["normalized_energy"], dtype=torch.float32)
        normalized_duration = torch.tensor(row["normalized_duration"], dtype=torch.float32)

        # Emotion embedding (one-hot encoding)
        emotion_label = self.label_encoder.transform([row['emotion']])[0]
        emotion_embedding = torch.tensor(emotion_label, dtype=torch.long)

        return {
            'text': text_tensor,
            'pitch': normalized_pitch,
            'energy': normalized_energy,
            'duration': normalized_duration,
            'emotion': emotion_embedding
        }
    
    def text_to_sequence(self, text):
        # Replace this with a proper text tokenizer or sequence generator.
        # Here we use a simple character-to-index mapping as an example.
        vocab = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        sequence = [vocab[char] for char in text.lower() if char in vocab]
        return sequence[:self.max_text_len]  # Truncate to max length

class CustomCollator:
    def __call__(self, batch):
        # Tokenize each text input in the batch and include prosodic features
        texts = [item['text'] for item in batch]
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Stack prosodic features and emotion labels if they are included
        pitch = torch.stack([item['avg_pitch'] for item in batch])
        energy = torch.stack([item['avg_energy'] for item in batch])
        duration = torch.stack([item['duration'] for item in batch])
        emotion_labels = torch.tensor([item['emotion'] for item in batch])

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "emotion_labels": emotion_labels
        }

if __name__ == "__main__":
    df = pd.read_csv("base_data_normalized.csv")

    label_encoder = LabelEncoder()
    df["emotion_encoded"] = label_encoder.fit_transform(df["emotion"])

    dataset = EmotionalTTSDataset(df, label_encoder)
    data_collator = CustomCollator()
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)

    # Freeze parts of the model if needed, especially at the start
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers for fine-tuning
    for param in model.decoder.parameters():
        param.requires_grad = True

    # Define optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    # Learning rate scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Extract inputs from batch
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            prosody_features = torch.stack([batch["pitch"], batch["energy"], batch["duration"]], dim=-1)
            emotion_labels = batch["emotion_labels"]

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prosody_features=prosody_features,
                emotion_labels=emotion_labels
            )

            # Calculate loss and backpropagate
            loss = outputs.loss  # Assuming the model has a loss attribute
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")