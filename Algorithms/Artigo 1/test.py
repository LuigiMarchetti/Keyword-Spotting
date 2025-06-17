import librosa
import numpy as np
import json
import os
from pathlib import Path
from joblib import load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações - DEVEM SER IDÊNTICAS AO TREINO (baseado no VoiceTrainer)
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = SCRIPT_DIR / '../../files/models/Artigo 1'  # Ajustado para o caminho correto
TEST_PATH = SCRIPT_DIR / '../../files/dataset'
TEST_PATH_RECORDED = SCRIPT_DIR / '../../files/recorded'
SAMPLES = ['ari', 'luigi']
WORDS = ['backward', 'forward', 'left', 'right', 'stop']  # Ordem correta do VoiceTrainer
SAMPLE_RATE = 11000  # Mesmo valor do VoiceTrainer

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
    commands = ['backward', 'forward', 'left', 'right', 'stop']  # Ordem padrão do VoiceTrainer
    print("AVISO: Usando mapeamento padrão")

# Audio preprocessing - DEVE SER IDÊNTICA AO VoiceTrainer
def preprocess_audio(signal, sr):
    """
    Normaliza e remove silêncios do sinal de áudio.
    IDÊNTICA ao método preprocess_audio do VoiceTrainer.
    """
    signal = librosa.util.normalize(signal)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    return signal

def extract_features(signal, sr):
    """
    Extrai os coeficientes MFCC e calcula a média.
    IDÊNTICA ao método extract_features do VoiceTrainer.
    """
    n_mfcc = 20  # Mesmo valor do VoiceTrainer
    n_fft = 256  # Mesmo valor do VoiceTrainer
    hop_length = 128  # Mesmo valor do VoiceTrainer

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def process_audio_file(file_path):
    """
    Processa um arquivo de áudio seguindo a mesma lógica do VoiceTrainer.
    """
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Mesma validação do VoiceTrainer
        if len(signal) < 0.1 * sr:
            return None

        signal = preprocess_audio(signal, sr)
        features = extract_features(signal, sr)
        return features
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

# Load model
try:
    model = load(MODEL_PATH / "svm_model.joblib")
    print("Modelo SVM carregado com sucesso!")

    # Verificar se o modelo é um pipeline (como no VoiceTrainer)
    if hasattr(model, 'named_steps'):
        print("Pipeline detectado com etapas:", list(model.named_steps.keys()))

    # Verificar suporte a probabilidades
    if not hasattr(model, 'predict_proba'):
        print("AVISO: Modelo não suporta probabilidades nativamente. Usando calibração.")
        # Para modelos SVM sem probabilidade, usar calibração
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        use_calibrated = True
    else:
        calibrated_model = model
        use_calibrated = False

except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

def predict_with_details(file_path):
    """
    Faz predição com detalhes de probabilidade.
    """
    features = process_audio_file(file_path)
    if features is None:
        print(f"Arquivo {os.path.basename(file_path)} muito curto ou com erro - ignorando")
        return None, None

    # Reshape para formato esperado pelo modelo (1 amostra)
    features = features.reshape(1, -1)

    # Predição
    predicted_idx = model.predict(features)[0]

    # Cálculo de probabilidades
    try:
        if use_calibrated:
            probabilities = calibrated_model.predict_proba(features)[0]
        elif hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
        else:
            # Fallback usando decision_function
            decision_values = model.decision_function(features)[0]
            # Converter para probabilidades usando softmax
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

# ========================
# PARTE 1 - TESTE K-FOLD
# ========================
print("\n=========== TESTE K-FOLD COM BASE DO TREINAMENTO ===========")
X_kf = []
y_kf = []

for label in WORDS:
    dir_path = TEST_PATH / label
    if not dir_path.exists():
        print(f"Diretório não encontrado: {dir_path}")
        continue

    wav_files = list(dir_path.glob("*.wav"))
    print(f"Processando {len(wav_files)} arquivos para '{label}'")

    for file in wav_files:
        features = process_audio_file(file)
        if features is not None:
            X_kf.append(features)
            y_kf.append(label)

X_kf = np.array(X_kf)
y_kf = np.array(y_kf)
print(f"Total de amostras válidas: {len(y_kf)}")

if len(y_kf) > 0:
    # K-Fold Cross Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X_kf, y_kf), 1):
        print(f"\n--- Fold {i} ---")
        X_test = X_kf[test_idx]
        y_test = y_kf[test_idx]

        # Predição usando o modelo carregado
        y_pred_indices = model.predict(X_test)
        y_pred = [commands[idx] if isinstance(idx, (int, np.integer)) else idx for idx in y_pred_indices]

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        # Acurácia do fold
        fold_accuracy = np.mean([t == p for t, p in zip(y_test, y_pred)])
        print(f"Acurácia Fold {i}: {fold_accuracy:.2%}")

    # Relatório final K-Fold
    print("\n=== RELATÓRIO GERAL - K-FOLD ===")
    print(classification_report(y_true_all, y_pred_all, labels=WORDS))

    # Matriz de Confusão
    cm = confusion_matrix(y_true_all, y_pred_all, labels=WORDS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=WORDS, yticklabels=WORDS, cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão - K-Fold")
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum dado válido encontrado para teste K-Fold!")

# ========================
# PARTE 2 - TESTE CASEIRO
# ========================
print("\n=========== TESTE COM AMOSTRAS CASEIRAS ===========")
total_correct = 0
total_files = 0

for speaker in SAMPLES:
    speaker_dir = TEST_PATH_RECORDED / speaker
    if not speaker_dir.exists():
        print(f"Diretório não encontrado: {speaker_dir}")
        continue

    print(f"\n=== Testando arquivos de {speaker} ===")
    wav_files = list(speaker_dir.glob('*.wav'))
    speaker_correct = 0
    valid_files = 0

    for wav_file in wav_files:
        # Extrair label esperado do nome do arquivo
        expected_label = wav_file.stem.split('_')[0]

        # Verificar se o label está nos comandos treinados
        if expected_label not in [cmd for cmd in commands if cmd]:
            print(f"Pulando {wav_file.name} - label '{expected_label}' não foi treinado")
            continue

        try:
            label, confidence = predict_with_details(wav_file)
            if label is None:
                continue

            is_correct = label == expected_label
            if is_correct:
                speaker_correct += 1
                total_correct += 1

            total_files += 1
            valid_files += 1

            status = '✓' if is_correct else '✗'
            print(f"Predição: {label} ({confidence*100:.2f}%) | Esperado: {expected_label} {status}")

        except Exception as e:
            print(f"Erro em {wav_file.name}: {str(e)}")

    # Relatório por speaker
    if valid_files > 0:
        accuracy = speaker_correct / valid_files * 100
        print(f"Acurácia para {speaker}: {speaker_correct}/{valid_files} = {accuracy:.2f}%")
    else:
        print(f"Nenhum arquivo válido para {speaker}")

# Relatório final
if total_files > 0:
    final_accuracy = total_correct / total_files * 100
    print(f"\n=== RESULTADO FINAL - CASEIRO ===")
    print(f"Acurácia geral: {total_correct}/{total_files} = {final_accuracy:.2f}%")
else:
    print("Nenhum arquivo válido testado!")

print("\n=== INFORMAÇÕES DO MODELO ===")
print(f"Comandos reconhecidos: {commands}")
print(f"Configurações de áudio:")
print(f"  - Sample rate: {SAMPLE_RATE} Hz")
print(f"  - Características MFCC: 20 coeficientes médios")
print(f"  - n_fft: 256, hop_length: 128")