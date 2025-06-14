import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class VoiceCommandRecognition:
    def __init__(self, sample_rate=11000, duration=2.0, n_mfcc=20):
        """
        Initialize the Voice Command Recognition system

        Parameters:
        - sample_rate: Sampling frequency (Hz)
        - duration: Recording duration (seconds)
        - n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.frame_length = 256
        self.commands = ['forward', 'backward', 'left', 'right', 'stop']
        self.svm_model = None
        self.scaler = StandardScaler()
        self.training_data = []
        self.training_labels = []

    def record_audio(self):
        """
        Record audio from microphone

        Returns:
        - audio_data: Recorded audio signal
        """
        print(f"Recording for {self.duration} seconds...")
        audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        print("Recording completed!")
        return audio_data.flatten()

    def preprocess_audio(self, audio_data):
        """
        Preprocess audio signal - remove silence and normalize

        Parameters:
        - audio_data: Raw audio signal

        Returns:
        - processed_audio: Preprocessed audio signal
        """
        # Remove silence using energy-based voice activity detection
        frame_size = 1024
        hop_size = 512

        # Calculate energy for each frame
        energy = []
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            frame_energy = np.sum(frame ** 2)
            energy.append(frame_energy)

        # Find voice activity regions
        energy = np.array(energy)
        energy_threshold = np.mean(energy) * 0.1

        # Find start and end of voice activity
        voice_frames = np.where(energy > energy_threshold)[0]
        if len(voice_frames) > 0:
            start_frame = voice_frames[0]
            end_frame = voice_frames[-1]

            # Convert frame indices to sample indices
            start_sample = start_frame * hop_size
            end_sample = min((end_frame + 1) * hop_size + frame_size, len(audio_data))

            processed_audio = audio_data[start_sample:end_sample]
        else:
            processed_audio = audio_data

        # Normalize audio
        if np.max(np.abs(processed_audio)) > 0:
            processed_audio = processed_audio / np.max(np.abs(processed_audio))

        return processed_audio

    def extract_mfcc_features(self, audio_data):
        """
        Extract MFCC features from audio signal

        Parameters:
        - audio_data: Audio signal

        Returns:
        - mfcc_features: MFCC feature vector
        """
        # Apply Hamming window and extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length * 2,
            hop_length=self.frame_length // 2,
            window='hamming'
        )

        # Calculate statistics (mean and std) of MFCC coefficients
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Combine mean and std to create feature vector
        mfcc_features = np.concatenate([mfcc_mean, mfcc_std])

        return mfcc_features

    def collect_training_data(self, samples_per_command=10):
        """
        Collect training data for all commands

        Parameters:
        - samples_per_command: Number of samples to collect per command
        """
        print("=== Training Data Collection ===")

        for command in self.commands:
            print(f"\nCollecting samples for command: '{command}'")
            print(f"Say '{command}' when prompted...")

            for i in range(samples_per_command):
                input(f"Press Enter and say '{command}' (Sample {i+1}/{samples_per_command}): ")

                # Record audio
                audio_data = self.record_audio()

                # Preprocess audio
                processed_audio = self.preprocess_audio(audio_data)

                # Extract MFCC features
                mfcc_features = self.extract_mfcc_features(processed_audio)

                # Store training data
                self.training_data.append(mfcc_features)
                self.training_labels.append(command)

                print(f"Sample {i+1} collected successfully!")

        print(f"\nTraining data collection completed!")
        print(f"Total samples: {len(self.training_data)}")

    def train_svm_model(self):
        """
        Train SVM model using collected training data
        """
        if len(self.training_data) == 0:
            raise ValueError("No training data available. Please collect training data first.")

        print("\n=== Training SVM Model ===")

        # Convert to numpy arrays
        X = np.array(self.training_data)
        y = np.array(self.training_labels)

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train SVM with RBF kernel (as mentioned in the paper)
        # Using one-vs-rest approach for multi-class classification
        self.svm_model = SVC(
            kernel='rbf',
            gamma=1000,  # Kernel parameter as mentioned in paper
            C=1.0,
            decision_function_shape='ovr',  # One-vs-rest
            random_state=42
        )

        self.svm_model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Training completed!")
        print(f"Validation Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return accuracy

    def predict_command(self, audio_data=None):
        """
        Predict voice command from audio data

        Parameters:
        - audio_data: Audio signal (if None, will record from microphone)

        Returns:
        - predicted_command: Predicted command
        - confidence: Prediction confidence
        """
        if self.svm_model is None:
            raise ValueError("Model not trained. Please train the model first.")

        if audio_data is None:
            audio_data = self.record_audio()

        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_data)

        # Extract MFCC features
        mfcc_features = self.extract_mfcc_features(processed_audio)

        # Normalize features
        mfcc_features_scaled = self.scaler.transform([mfcc_features])

        # Predict
        predicted_command = self.svm_model.predict(mfcc_features_scaled)[0]

        # Get prediction confidence (distance from decision boundary)
        decision_scores = self.svm_model.decision_function(mfcc_features_scaled)
        confidence = np.max(decision_scores)

        return predicted_command, confidence

    def save_model(self, filename='voice_command_model.pkl'):
        """
        Save trained model and scaler

        Parameters:
        - filename: File name to save the model
        """
        if self.svm_model is None:
            raise ValueError("No trained model to save.")

        model_data = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'commands': self.commands,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'n_mfcc': self.n_mfcc
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filename}")

    def load_model(self, filename='voice_command_model.pkl'):
        """
        Load trained model and scaler

        Parameters:
        - filename: File name to load the model from
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found.")

        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        self.svm_model = model_data['svm_model']
        self.scaler = model_data['scaler']
        self.commands = model_data['commands']
        self.sample_rate = model_data['sample_rate']
        self.duration = model_data['duration']
        self.n_mfcc = model_data['n_mfcc']

        print(f"Model loaded from {filename}")

    def visualize_mfcc_patterns(self):
        """
        Visualize MFCC patterns for different commands
        """
        if len(self.training_data) == 0:
            print("No training data available for visualization.")
            return

        # Group data by command
        command_data = {cmd: [] for cmd in self.commands}

        for features, label in zip(self.training_data, self.training_labels):
            command_data[label].append(features)

        # Plot MFCC patterns
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, command in enumerate(self.commands):
            if command_data[command]:
                # Calculate mean MFCC pattern for each command
                mean_mfcc = np.mean(command_data[command], axis=0)
                axes[i].plot(mean_mfcc[:self.n_mfcc], 'b-', linewidth=2, label='Mean')
                axes[i].plot(mean_mfcc[self.n_mfcc:], 'r--', linewidth=2, label='Std')
                axes[i].set_title(f'MFCC Pattern - {command.upper()}')
                axes[i].set_xlabel('MFCC Coefficient')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True)

        # Remove extra subplot
        if len(self.commands) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.show()

    def test_recognition(self, num_tests=10):
        """
        Test the recognition system with real-time audio

        Parameters:
        - num_tests: Number of test iterations
        """
        if self.svm_model is None:
            raise ValueError("Model not trained. Please train the model first.")

        print("\n=== Testing Voice Command Recognition ===")
        print("Available commands:", ', '.join(self.commands))

        correct_predictions = 0

        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests}")
            expected_command = input("Enter the command you will say: ").lower().strip()

            if expected_command not in self.commands:
                print("Invalid command. Please use one of:", ', '.join(self.commands))
                continue

            input("Press Enter and say the command: ")

            try:
                predicted_command, confidence = self.predict_command()
                print(f"Expected: {expected_command}")
                print(f"Predicted: {predicted_command}")
                print(f"Confidence: {confidence:.2f}")

                if predicted_command == expected_command:
                    print("✓ CORRECT!")
                    correct_predictions += 1
                else:
                    print("✗ INCORRECT!")

            except Exception as e:
                print(f"Error during prediction: {e}")

        accuracy = correct_predictions / num_tests
        print(f"\nTest Results:")
        print(f"Correct: {correct_predictions}/{num_tests}")
        print(f"Accuracy: {accuracy:.2%}")

def main():
    """
    Main function to demonstrate the voice command recognition system
    """
    print("Voice Command Recognition System")
    print("Based on MFCC and SVM (Implementation of the research paper)")
    print("="*60)

    # Initialize the system
    vcr = VoiceCommandRecognition()

    while True:
        print("\nOptions:")
        print("1. Collect training data")
        print("2. Train SVM model")
        print("3. Test recognition")
        print("4. Save model")
        print("5. Load model")
        print("6. Visualize MFCC patterns")
        print("7. Single prediction")
        print("8. Exit")

        choice = input("\nEnter your choice (1-8): ").strip()

        try:
            if choice == '1':
                samples = int(input("Enter number of samples per command (default 10): ") or "10")
                vcr.collect_training_data(samples_per_command=samples)

            elif choice == '2':
                vcr.train_svm_model()

            elif choice == '3':
                tests = int(input("Enter number of tests (default 10): ") or "10")
                vcr.test_recognition(num_tests=tests)

            elif choice == '4':
                filename = input("Enter filename to save (default: voice_command_model.pkl): ").strip()
                if not filename:
                    filename = 'voice_command_model.pkl'
                vcr.save_model(filename)

            elif choice == '5':
                filename = input("Enter filename to load (default: voice_command_model.pkl): ").strip()
                if not filename:
                    filename = 'voice_command_model.pkl'
                vcr.load_model(filename)

            elif choice == '6':
                vcr.visualize_mfcc_patterns()

            elif choice == '7':
                if vcr.svm_model is None:
                    print("Please train or load a model first.")
                else:
                    print("Say a command...")
                    predicted_command, confidence = vcr.predict_command()
                    print(f"Predicted command: {predicted_command}")
                    print(f"Confidence: {confidence:.2f}")

            elif choice == '8':
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()