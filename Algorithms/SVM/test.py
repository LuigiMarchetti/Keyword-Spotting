import librosa
import numpy as np
import json
import os
from pathlib import Path
from joblib import load
from sklearn.calibration import CalibratedClassifierCV

# Configurações - DEVEM SER IDÊNTICAS AO TREINO
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / '../../files/models/SVM/'
TEST_PATH = SCRIPT_DIR / '../../files/recorded'
SAMPLES = ['ari', 'luigi']
SAMPLE_RATE = 11000
AUDIO_LENGTH = 1.0
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH)

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
    """Garante 1 segundo de áudio - IDÊNTICA AO TREINO SVM"""
    # Normalização
    signal = librosa.util.normalize(signal)

    # Remoção de silêncio
    signal, _ = librosa.effects.trim(signal, top_db=20)

    # Padding/truncamento
    if len(signal) > N_SAMPLES:
        signal = signal[:N_SAMPLES]
    else:
        signal = np.pad(signal, (0, N_SAMPLES - len(signal)), 'constant')

    return signal

def extract_features(signal, sr):
    """DEVE SER IDÊNTICA AO TREINO SVM"""
    # Parâmetros do MFCC (devem ser iguais ao treino)
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13

    # MFCCs
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)

    # Derivadas dos MFCCs (delta e delta-delta)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Features espectrais adicionais
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)[0]

    # Estatísticas das features
    features = []

    # MFCC stats
    features.extend([mfcc.mean(axis=1), mfcc.std(axis=1),
                     mfcc.max(axis=1), mfcc.min(axis=1)])

    # Delta MFCC stats
    features.extend([mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1)])

    # Delta-delta MFCC stats
    features.extend([mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1)])

    # Spectral features stats
    features.extend([
        [spectral_centroids.mean(), spectral_centroids.std()],
        [spectral_rolloff.mean(), spectral_rolloff.std()],
        [zero_crossing_rate.mean(), zero_crossing_rate.std()]
    ])

    return np.concatenate([f.flatten() if hasattr(f, 'flatten') else f for f in features])

def process_audio_file(file_path):
    """Carrega e processa um arquivo de áudio"""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pular arquivos muito curtos
    if len(signal) < 0.1 * sr:  # Menos que 0.1 segundo
        return None

    signal = preprocess_audio(signal)
    features = extract_features(signal, sr)
    return features

# Load model
try:
    model = load(MODEL_PATH / "svm_model.joblib")
    print("Modelo SVM carregado com sucesso!")

    # Verifica se o modelo suporta probabilidades
    if not hasattr(model, 'predict_proba'):
        print("AVISO: Modelo não suporta probabilidades nativamente. Usando aproximação.")
        # Cria um classificador calibrado para estimar probabilidades
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        # Note: Você precisaria ter alguns dados de validação para calibrar corretamente
        # Isso é apenas uma aproximação básica
        print("AVISO: As probabilidades são aproximadas e podem não ser precisas!")
    else:
        calibrated_model = model

except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

def predict_with_details(file_path):
    features = process_audio_file(file_path)
    if features is None:
        print(f"Arquivo {os.path.basename(file_path)} muito curto - ignorando")
        return None, None

    # Obter a classe predita
    predicted_idx = model.predict([features])[0]

    # Obter probabilidades (ou aproximação)
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([features])[0]
        else:
            # Aproximação básica quando não há predict_proba
            decision_values = model.decision_function([features])[0]
            probabilities = np.exp(decision_values) / np.sum(np.exp(decision_values))
    except Exception as e:
        print(f"Erro ao calcular probabilidades: {e}")
        probabilities = np.zeros(len(commands))
        probabilities[predicted_idx] = 1.0

    print(f"\nArquivo: {os.path.basename(file_path)}")
    print("Probabilidades estimadas para cada classe:")
    for i, prob in enumerate(probabilities):
        print(f"  {commands[i]}: {prob*100:.2f}%")

    return commands[predicted_idx], probabilities[predicted_idx]

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
    valid_files = 0

    for wav_file in wav_files:
        expected_label = wav_file.stem.split('_')[0]

        # Pular arquivos que não estão no mapeamento de treino
        if expected_label not in [cmd for cmd in commands if cmd]:
            print(f"Pulando {wav_file.name} - label '{expected_label}' não foi treinado")
            continue

        try:
            label, confidence = predict_with_details(wav_file)
            if label is None:  # Arquivo muito curto
                continue

            is_correct = label == expected_label
            if is_correct:
                speaker_correct += 1
                total_correct += 1
            total_files += 1
            valid_files += 1

            print(f"Predição: {label} ({confidence*100:.2f}%) | Esperado: {expected_label} {'✓' if is_correct else '✗'}")
        except Exception as e:
            print(f"Erro em {wav_file.name}: {str(e)}")

    if valid_files > 0:
        print(f"Acurácia para {speaker}: {speaker_correct}/{valid_files} = {speaker_correct/max(1,valid_files)*100:.2f}%")
    else:
        print(f"Nenhum arquivo válido para {speaker}")

if total_files > 0:
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Acurácia geral: {total_correct}/{total_files} = {total_correct/total_files*100:.2f}%")
else:
    print("Nenhum arquivo válido testado!")