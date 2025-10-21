import numpy as np

feat = np.load('/Users/rritahajrizi/BCI-Brain2Speech/data/processed/features/sub-01_feat.npy')
melSpec = np.load('/Users/rritahajrizi/BCI-Brain2Speech/data/processed/features/sub-01_spec.npy')
words = np.load('/Users/rritahajrizi/BCI-Brain2Speech/data/processed/features/sub-01_procWords.npy', allow_pickle=True)

print("EEG features shape:", feat.shape)
print("Mel spectrogram shape:", melSpec.shape)
print("Words shape:", words.shape)
