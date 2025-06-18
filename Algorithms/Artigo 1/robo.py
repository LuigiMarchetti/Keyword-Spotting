import sounddevice as sd
import numpy as np
import librosa
import joblib
import json
import time
from pathlib import Path

# Parâmetros
DURATION = 2  # segundos de escuta
SAMPLE_RATE = 16000
COMMANDS = ["left", "right", "forward", "backward", "stop"]
CONFIDENCE_THRESHOLD = 0.3  # 30% de confiança mínima
IS_MANUAL_MODE = True

# Carrega o melhor modelo e o seu mapeamento
model = joblib.load("../../files/models/SVM/svm_model.joblib")
with open("../../files/models/SVM/label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Inverter mapeamento (de índice para label)
idx_to_label = {v: k for k, v in label_mapping.items()}


def contains_voice(audio, threshold=0.02):
    """Verifica se o áudio tem voz com base na energia RMS (antes do processamento)"""
    rms = np.sqrt(np.mean(audio ** 2))
    return rms > threshold


def preprocess_audio(signal, sr):
    """Normaliza, remove silêncio e ajusta o tamanho"""
    signal = librosa.util.normalize(signal)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    target_length = sr  # 1 segundo
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
    return signal


def extract_features(signal, sr):
    """Extrai as mesmas features usadas no treinamento"""
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)[0]

    features = []
    features.extend([mfcc.mean(axis=1), mfcc.std(axis=1),
                     mfcc.max(axis=1), mfcc.min(axis=1)])
    features.extend([mfcc_delta.mean(axis=1), mfcc_delta.std(axis=1)])
    features.extend([mfcc_delta2.mean(axis=1), mfcc_delta2.std(axis=1)])
    features.extend([
        [spectral_centroids.mean(), spectral_centroids.std()],
        [spectral_rolloff.mean(), spectral_rolloff.std()],
        [zero_crossing_rate.mean(), zero_crossing_rate.std()]
    ])

    return np.concatenate([f.flatten() if hasattr(f, 'flatten') else f for f in features]).reshape(1, -1)


def execute_command(command, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"❓ Comando não reconhecido (confiança: {confidence:.1%})")
        return

    print(f"🔊 Comando detectado: {command} (confiança: {confidence:.1%})")
    if command == "left":
        print("🤖 Virando para a ESQUERDA")
    elif command == "right":
        print("🤖 Virando para a DIREITA")
    elif command == "forward":
        print("🤖 Indo PARA FRENTE")
    elif command == "backward":
        print("🤖 Recuando PARA TRÁS")
    elif command == "stop":
        print("⛔ Parando movimento e encerrando o programa")
    else:
        print("❓ Comando não reconhecido")


def listen_and_predict(manual_mode=False):
    """Função principal para capturar áudio e prever comandos"""
    if manual_mode:
        input("Pressione Enter para começar a gravar...")
        print("Gravando...")

    # Capturar áudio
    audio = None
    while True:
        buffer = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        buffer = buffer.flatten()

        if contains_voice(buffer, threshold=0.01):
            audio = buffer
            break
        elif manual_mode:
            return None  # Não detectou voz no modo manual
        else:
            time.sleep(0.5)

    # Processar áudio
    audio = preprocess_audio(audio, SAMPLE_RATE)
    feat = extract_features(audio, SAMPLE_RATE)

    # Prever comando e probabilidades (se disponível)
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(feat)[0]
        prediction = np.argmax(probas)
        confidence = probas[prediction]
    else:
        prediction = model.predict(feat)[0]
        confidence = 1.0  # Assume 100% de confiança se o modelo não fornecer probabilidades

    predicted_label = idx_to_label.get(prediction, "unknown")

    return predicted_label, confidence


def main(manual_mode=False):
    print("Robô iniciado. Fale um comando: left, right, forward, backward, stop")
    print("Diga 'stop' para encerrar.\n")

    try:
        while True:
            if manual_mode:
                print("\nModo manual ativado - pressione Enter para gravar ou 'q' para sair")
                user_input = input()
                if user_input.lower() == 'q':
                    break

            result = listen_and_predict(manual_mode)
            if result is None:
                continue  # Não detectou voz no modo manual

            command, confidence = result
            execute_command(command, confidence)
            print("-" * 40)

            if command == "stop":
                break

            if not manual_mode:
                time.sleep(1)

    except KeyboardInterrupt:
        print("Encerrando escuta manualmente...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manual', action='store_true', help='Ativar modo manual (pressionar Enter para gravar)')
    args = parser.parse_args()

    main(manual_mode=IS_MANUAL_MODE)