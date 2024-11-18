
# Text Emotion Prediction

## Overview
This module focuses on building an emotion prediction model for the Kazakh language. The model is trained using a fine-tuned version of the `ai-forever/mGPT-1.3B-Kazakh` model to predict emotions based on textual input. Due to the limited availability of dedicated emotion-labeled data for Kazakh, a text-to-speech (TTS) dataset (`KazEmoTTS`) was adapted for this purpose, presenting unique challenges and insights.

## Folder Structure
```
Text Emotion Prediction/
├── data/
│   ├── emotion_text_data.csv         # Adapted dataset for emotion prediction
├── preprocess_data.py                # Script for text preprocessing
├── train_model.py                    # Script for model training and evaluation
├── results/
│   ├── training_validation_loss.png  # Graph showing training and validation loss
│   ├── confusion_matrix.png          # Confusion matrix visualization
└── README.md                         # Documentation for the Text Emotion Prediction module
```

## Requirements
- Python 3.8 or higher
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- Pandas
- Matplotlib

To install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
The dataset used for this task is adapted from a TTS dataset. The preprocessing step involves cleaning and transforming the data for emotion prediction tasks. The six emotion categories are: `neutral`, `surprise`, `fear`, `angry`, `happy`, and `sad`.

Run the preprocessing script:
```bash
python preprocess_data.py
```

## Model Training
The `ai-forever/mGPT-1.3B-Kazakh` model was fine-tuned using the adapted dataset. Training is managed through PyTorch and Hugging Face's Trainer API.

### Hyperparameters Used
- **Learning Rate**: 5e-5
- **Batch Size**: 16
- **Epochs**: 30

To train the model, execute:
```bash
python train_model.py
```

## Results
### Training and Validation Loss
The graph below illustrates the training and validation loss over 30 epochs.

![Training and Validation Loss](results/loss.png)

### Confusion Matrix
The confusion matrix highlights the challenges faced during classification due to data constraints.

![Confusion Matrix](results/conf_table.png)

## Evaluation Metrics
- **Accuracy**: Used to evaluate the overall correctness of predictions.
- **Confusion Matrix**: Provides a breakdown of correct and incorrect classifications for each category.
- **Precision, Recall, and F1-Score**: Offers detailed performance evaluation for each emotion class.

## Limitations and Challenges
The primary limitation was the adaptation of a TTS dataset for emotion prediction, leading to potential data mismatches and reduced performance. This highlights the need for a more specialized dataset for accurate emotion prediction in Kazakh.

## Future Directions
- Acquisition or creation of a dedicated Kazakh emotion prediction dataset.
- Incorporation of advanced data augmentation techniques.
- Exploration of alternative model architectures for improved performance.
