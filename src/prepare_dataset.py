import os
import numpy as np
from sklearn.model_selection import train_test_split

# config
features_dir = '/Users/rritahajrizi/BCI-Brain2Speech/data/processed/features'
participants = ['sub-01']

# load and concaterate data
all_feat = []
all_mel = []
all_words = []

for p in participants:
    feat_path = os.path.join(features_dir, f'{p}_feat.npy')
    mel_path = os.path.join(features_dir, f'{p}_spec.npy')
    words_path = os.path.join(features_dir, f'{p}_procWords.npy')

    if not os.path.exists(feat_path):
        print(f" Skipping {p}: missing feature file.")
        continue

    print(f"Loading data for {p}...")

    feat = np.load(feat_path)
    melSpec = np.load(mel_path)
    words = np.load(words_path, allow_pickle=True)  # required for string arrays

    # Optional sanity check — align lengths
    min_len = min(feat.shape[0], melSpec.shape[0], words.shape[0])
    feat, melSpec, words = feat[:min_len], melSpec[:min_len], words[:min_len]

    all_feat.append(feat)
    all_mel.append(melSpec)
    all_words.append(words)

# Combine all subjects
X = np.vstack(all_feat)          # EEG features
y_audio = np.vstack(all_mel)     # Mel spectrograms (for regression)
y_words = np.concatenate(all_words)  # Word labels (optional classification)

print(" Combined dataset shapes:")
print("EEG features:", X.shape)
print("Mel spectrograms:", y_audio.shape)
print("Word labels:", y_words.shape)

# split into train/val/test set 
# 70% train, 15% validation, 15% test
X_train, X_temp, y_audio_train, y_audio_temp, y_words_train, y_words_temp = train_test_split(
    X, y_audio, y_words, test_size=0.3, random_state=42, shuffle=True
)

X_val, X_test, y_audio_val, y_audio_test, y_words_val, y_words_test = train_test_split(
    X_temp, y_audio_temp, y_words_temp, test_size=0.5, random_state=42, shuffle=True
)

print("\nData split:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# save datasets
os.makedirs(os.path.join(features_dir, 'splits'), exist_ok=True)

np.save(os.path.join(features_dir, 'splits/X_train.npy'), X_train)
np.save(os.path.join(features_dir, 'splits/X_val.npy'), X_val)
np.save(os.path.join(features_dir, 'splits/X_test.npy'), X_test)

np.save(os.path.join(features_dir, 'splits/y_audio_train.npy'), y_audio_train)
np.save(os.path.join(features_dir, 'splits/y_audio_val.npy'), y_audio_val)
np.save(os.path.join(features_dir, 'splits/y_audio_test.npy'), y_audio_test)

np.save(os.path.join(features_dir, 'splits/y_words_train.npy'), y_words_train)
np.save(os.path.join(features_dir, 'splits/y_words_val.npy'), y_words_val)
np.save(os.path.join(features_dir, 'splits/y_words_test.npy'), y_words_test)

print("\n Saved all split datasets in:", os.path.join(features_dir, 'splits'))
