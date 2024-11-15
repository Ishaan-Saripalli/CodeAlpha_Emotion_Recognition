from pydub import AudioSegment
import librosa
import numpy as np
import os

def convert_m4a_to_wav(m4a_file):
    """Convert .m4a file to .wav format."""
    wav_file = m4a_file.replace('.m4a', '.wav')
    audio = AudioSegment.from_file(m4a_file, format="m4a")
    audio.export(wav_file, format="wav")
    return wav_file

def extract_features(audio_file):
    """Extract MFCC features from an audio file."""
    if audio_file.endswith('.m4a'):
        audio_file = convert_m4a_to_wav(audio_file)
    y, sr = librosa.load(audio_file, duration=2.5, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def load_data(audio_directory):
    """Load features and labels from all audio files in a directory."""
    features = []
    labels = []
    for filename in os.listdir(audio_directory):
        if filename.endswith(".m4a"):
            file_path = os.path.join(audio_directory, filename)
            feature = extract_features(file_path)
            features.append(feature)
        
            label = filename.split("-")[2]  
            labels.append(label)
    return np.array(features), np.array(labels)
audio_directory = r'C:\Users\DELL\Desktop\Emotion Recognition from Speech'

# Load data
X, y = load_data(audio_directory)

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)
