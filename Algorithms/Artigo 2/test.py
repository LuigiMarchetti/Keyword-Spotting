import torch
import librosa
import numpy as np
import torch.nn as nn
import json
import os
from pathlib import Path

# Configurações - DEVEM SER IDÊNTICAS AO TREINO
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / '../../files/models/Artigo 2/'
TEST_PATH = SCRIPT_DIR / '../../files/recorded'
SAMPLES = ['ari', 'luigi']
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2.0
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model class - DEVE SER IDÊNTICA AO TREINO
class VoiceClassifier(nn.Module):
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

# Load label mapping
label_mapping_path = MODEL_PATH / "label_mapping.json"
if label_mapping_path.exists():
    with open(label_mapping_path, "r") as f:
        label_mapping = json.load(f)
    commands = [None] * len(label_mapping)
    for label, idx in label_mapping.items():
        commands[idx] = label
    print(f"Mapeamento carregado: {label_mapping}")
else:
    commands = ['stop', 'left', 'right', 'forward', 'backward']
    print("AVISO: Usando mapeamento padrão")

# Audio preprocessing - DEVE SER IDÊNTICA AO TREINO
def preprocess_audio(signal):
    """Garante 2 segundos de áudio - IDÊNTICA AO TREINO"""
    if len(signal) > N_SAMPLES:
        # Pega do centro
        start = (len(signal) - N_SAMPLES) // 2
        signal = signal[start:start + N_SAMPLES]
    elif len(signal) < N_SAMPLES:
        # Pad com zeros
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)), mode='constant')

    # Normaliza volume
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal)) * 0.8

    return signal

def extract_features(file_path):
    """DEVE SER IDÊNTICA AO TREINO"""
    # Carrega com SAMPLE_RATE fixo (não usar o sr retornado)
    signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    signal = preprocess_audio(signal)

    # MFCCs básicos - USAR SAMPLE_RATE FIXO
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=13)

    # Normalização simples - IDÊNTICA AO TREINO
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    # Flatten para entrada do MLP
    mfcc_flat = mfcc.flatten()

    return torch.tensor(mfcc_flat, dtype=torch.float32)

# Primeiro descobrir o input_size real
print("=== DESCOBRINDO INPUT SIZE ===")
test_files = list((TEST_PATH / SAMPLES[0]).glob('*.wav'))
if test_files:
    sample_features = extract_features(test_files[0])
    input_size = sample_features.shape[0]
    print(f"Input size detectado: {input_size}")
else:
    print("ERRO: Nenhum arquivo de teste encontrado!")
    exit()

# Load model
try:
    model = VoiceClassifier(input_size, len(commands)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH / "voice_model.pt", map_location=device))
    model.eval()
    print(f"Model loaded successfully com input_size={input_size}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def predict_with_details(file_path):
    features = extract_features(file_path).to(device)

    with torch.no_grad():
        outputs = model(features.unsqueeze(0))
        probs = torch.softmax(outputs, dim=1)

    print(f"\nArquivo: {os.path.basename(file_path)}")
    print("Probabilidades:")
    for i, (command, prob) in enumerate(zip(commands, probs[0])):
        print(f"  {command}: {prob.item()*100:.2f}%")

    predicted_idx = torch.argmax(probs, dim=1).item()
    return commands[predicted_idx], probs[0][predicted_idx].item()

# Test all files
total_correct = 0
total_files = 0

for speaker in SAMPLES:
    speaker_dir = TEST_PATH / speaker
    if not speaker_dir.exists():
        print(f"Diretório não encontrado: {speaker_dir}")
        continue

    print(f"\n=== Testando arquivos de {speaker} ===")

    wav_files = [f for f in speaker_dir.glob('*.wav')]
    speaker_correct = 0

    for wav_file in wav_files:
        expected_label = wav_file.stem.split('_')[0]

        # Pular arquivos que não estão no mapeamento de treino
        if expected_label not in [cmd for cmd in commands if cmd]:
            print(f"Pulando {wav_file.name} - label '{expected_label}' não foi treinado")
            continue

        try:
            label, confidence = predict_with_details(wav_file)
            is_correct = label == expected_label
            if is_correct:
                speaker_correct += 1
                total_correct += 1
            total_files += 1

            print(f"Predição: {label} ({confidence*100:.2f}%) | Esperado: {expected_label} {'✓' if is_correct else '✗'}")
        except Exception as e:
            print(f"Erro em {wav_file.name}: {str(e)}")

    print(f"Acurácia para {speaker}: {speaker_correct}/{len([f for f in wav_files if f.stem.split('_')[0] in commands])} = {speaker_correct/max(1,len([f for f in wav_files if f.stem.split('_')[0] in commands]))*100:.2f}%")

if total_files > 0:
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Acurácia geral: {total_correct}/{total_files} = {total_correct/total_files*100:.2f}%")
else:
    print("Nenhum arquivo válido testado!")