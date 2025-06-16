import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import KFold
import json

# Configuration - UPDATE THESE PATHS
MODEL_PATH = '../../files/models/Neural Network/simple_voice_model.pt'
DATASET_PATH = '../../files/dataset'
LABEL_MAPPING_PATH = '../../files/models/Neural Network/label_mapping.json'
commands = ['stop', 'left', 'right', 'forward', 'backward']
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2.0  # 2 seconds
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class (same as during training)
class SimpleAudioDataset(Dataset):
    def __init__(self, files, labels, label2idx, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment
        self.label2idx = label2idx

    def __len__(self):
        return len(self.files)

    def preprocess_audio(self, signal):
        if len(signal) > N_SAMPLES:
            start = (len(signal) - N_SAMPLES) // 2
            signal = signal[start:start + N_SAMPLES]
        elif len(signal) < N_SAMPLES:
            signal = np.pad(signal, (0, N_SAMPLES - len(signal)), mode='constant')

        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.8
        return signal

    def __getitem__(self, idx):
        signal, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        signal = self.preprocess_audio(signal)

        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        mfcc_flat = mfcc.flatten()

        return torch.tensor(mfcc_flat, dtype=torch.float32), self.label2idx[self.labels[idx]]

# Model class (must match training exactly)
class SimpleVoiceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy

def kfold_evaluation(model_path, n_splits=5):
    # Load label mapping
    with open(LABEL_MAPPING_PATH) as f:
        label2idx = json.load(f)

    # Collect all data
    X_paths, y_labels = [], []
    for label in commands:
        folder = os.path.join(DATASET_PATH, label)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
        X_paths.extend(files)
        y_labels.extend([label]*len(files))

    # Get input size from first sample
    sample_dataset = SimpleAudioDataset([X_paths[0]], [y_labels[0]], label2idx)
    input_size = sample_dataset[0][0].shape[0]

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_paths)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Create test dataset for this fold
        X_test = [X_paths[i] for i in test_idx]
        y_test = [y_labels[i] for i in test_idx]
        test_dataset = SimpleAudioDataset(X_test, y_test, label2idx)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Load model
        model = SimpleVoiceClassifier(input_size, len(commands)).to(device)
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        accuracy = evaluate_model(model, test_loader)
        fold_accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.2f}%")

    # Print final results
    print("\nFinal Results:")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.2f}%")
    print(f"Standard Deviation: {np.std(fold_accuracies):.2f}%")
    print("Individual Fold Accuracies:", [f"{acc:.2f}%" for acc in fold_accuracies])

    return fold_accuracies

if __name__ == "__main__":
    print("Starting K-Fold Evaluation...")
    print(f"Using device: {device}")

    accuracies = kfold_evaluation(MODEL_PATH)

    print("\nEvaluation complete!")