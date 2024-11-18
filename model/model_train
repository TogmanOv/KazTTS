import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Loading and preprocesing our dataset
df = pd.read_csv('emotion_text_data.csv')  
def preprocess_text(text):
    text = text.lower().replace("\n", " ")  # Basic preprocessing
    return text
df['Text'] = df['Text'].apply(preprocess_text)

# Splitting the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(" ai-forever/mGPT-1.3B-kazakh")

# Tokenizing the data with data augmentation 
def tokenize_function(texts):
    return tokenizer(
        texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt'
    )

train_encodings = tokenize_function(list(train_df['Text']))
val_encodings = tokenize_function(list(val_df['Text']))

# Converting labels to tensors (ensure the label mapping is consistent with our 6 classes but with starting 0 index)
label_mapping = {"neutral": 0, "sad": 1, "angry": 2, "fear": 3, "happy": 4, "surprise": 5}
train_labels = torch.tensor([label_mapping[label] for label in train_df['Emotion']])
val_labels = torch.tensor([label_mapping[label] for label in val_df['Emotion']])

# Preparing datasets for the Trainer API
from torch.utils.data import Dataset
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

# Loading the model with additional regularization layers 
model = AutoModelForSequenceClassification.from_pretrained(" ai-forever/mGPT-1.3B-kazakh", num_labels=6)

# Define a compute_metrics function for evaluation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments with improved regularization and learning schedule
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,  # Increased epochs for better learning
    weight_decay=0.01,  # Regularization
    learning_rate=5e-5,  # Fine-tuned learning rate
    warmup_steps=100,  # Gradual increase in learning rate during warmup
    logging_dir='./logs',
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

# Initializing the Trainer with gradient accumulation to handle large datasets better
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
trainer.save_model('./final_model')
