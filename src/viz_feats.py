import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def explore_and_visualize_features(participant='sub-01'):
    """Complete data exploration and visualization in one function"""

    # CORRECTED PATH
    base_path = './data/processed/features'
    
    # Load extracted features
    print("Loading features...")
    try:
        feat = np.load(f'{base_path}/{participant}_feat.npy')
        melSpec = np.load(f'{base_path}/{participant}_spec.npy')
        words = np.load(f'{base_path}/{participant}_procWords.npy', allow_pickle=True)
        feature_names = np.load(f'{base_path}/{participant}_feat_names.npy', allow_pickle=True)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("=== DATA EXPLORATION ===")
    print(f"EEG Features shape: {feat.shape}")
    print(f"Mel Spectrograms shape: {melSpec.shape}")
    print(f"Words shape: {words.shape}")
    print(f"Feature names shape: {feature_names.shape}")

    print("\n=== DATA STATISTICS ===")
    print(f"EEG Features - Min: {feat.min():.3f}, Max: {feat.max():.3f}, Mean: {feat.mean():.3f}, Std: {feat.std():.3f}")
    print(f"Mel Spectrograms - Min: {melSpec.min():.3f}, Max: {melSpec.max():.3f}, Mean: {melSpec.mean():.3f}, Std: {melSpec.std():.3f}")

    print("\n=== WORD DISTRIBUTION ===")
    unique_words, counts = np.unique(words, return_counts=True)
    word_df = pd.DataFrame({'Word': unique_words, 'Count': counts})
    print(word_df.sort_values('Count', ascending=False).head(10))

    # Create comprehensive visualizations
    print("\nCreating visualizations...")
    fig = plt.figure(figsize=(20, 15))

    # 1. EEG features timeline
    plt.subplot(3, 3, 1)
    plt.plot(feat[:500, 0])  # First 500 samples, first channel
    plt.title('EEG Feature Timeline\n(First Channel)')
    plt.xlabel('Time Windows')
    plt.ylabel('High-Gamma Power')
    plt.grid(True, alpha=0.3)

    # 2. Mel spectrogram
    plt.subplot(3, 3, 2)
    plt.imshow(melSpec[:100].T, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Mel Spectrogram\n(First 100 Frames)')
    plt.xlabel('Time Windows')
    plt.ylabel('Mel Bins')
    plt.colorbar(label='dB')

    # 3. Word distribution bar plot
    plt.subplot(3, 3, 3)
    plt.bar(range(len(unique_words)), counts)
    plt.title('Word Distribution')
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.xticks(range(len(unique_words)), unique_words, rotation=45, ha='right')
    plt.tight_layout()

    # 4. Feature correlation matrix (first 10 features)
    plt.subplot(3, 3, 4)
    corr_matrix = np.corrcoef(feat[:300, :10].T)
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('EEG Feature Correlation\n(First 10 Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.colorbar(im)

    # 5. Distribution of EEG feature values
    plt.subplot(3, 3, 5)
    plt.hist(feat[:, 0], bins=50, alpha=0.7, edgecolor='black')
    plt.title('EEG Feature Distribution\n(First Channel)')
    plt.xlabel('High-Gamma Power')
    plt.ylabel('Frequency')

    # 6. Distribution of Mel feature values
    plt.subplot(3, 3, 6)
    plt.hist(melSpec[:, 0], bins=50, alpha=0.7, edgecolor='black', color='orange')
    plt.title('Mel Feature Distribution\n(First Bin)')
    plt.xlabel('Mel Power (dB)')
    plt.ylabel('Frequency')

    # 7. Feature means across channels
    plt.subplot(3, 3, 7)
    feature_means = np.mean(feat, axis=0)
    # Plot first 50 features to avoid overcrowding
    plt.plot(feature_means[:50], 'o-', markersize=3)
    plt.title('Mean Feature Values\n(First 50 Features)')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.grid(True, alpha=0.3)

    # 8. Temporal evolution of a word
    plt.subplot(3, 3, 8)
    if len(unique_words) > 0:
        example_word = unique_words[0]
        word_indices = np.where(words == example_word)[0]
        if len(word_indices) > 0:
            # Plot EEG features around word occurrence
            example_idx = word_indices[0]
            start = max(0, example_idx - 10)
            end = min(len(feat), example_idx + 10)
            plt.plot(feat[start:end, 0])
            plt.axvline(x=10, color='red', linestyle='--', label='Word onset')
            plt.title(f'EEG around word: {example_word}')
            plt.xlabel('Time Windows')
            plt.ylabel('High-Gamma Power')
            plt.legend()

    # 9. Data quality check
    plt.subplot(3, 3, 9)
    # Check for NaN or Inf values
    eeg_nans = np.isnan(feat).sum()
    mel_nans = np.isnan(melSpec).sum()
    categories = ['EEG NaN', 'Mel NaN', 'EEG Inf', 'Mel Inf']
    values = [eeg_nans, mel_nans, np.isinf(feat).sum(), np.isinf(melSpec).sum()]
    plt.bar(categories, values, color=['red', 'red', 'blue', 'blue'])
    plt.title('Data Quality Check')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.suptitle(f'Feature Exploration - {participant}', fontsize=16, y=0.98)
    plt.show()

    # Print additional insights
    print("\n=== ADDITIONAL INSIGHTS ===")
    print(f"Total samples: {len(feat)}")
    print(f"EEG feature dimensionality: {feat.shape[1]}")
    print(f"Mel spectrogram dimensionality: {melSpec.shape[1]}")
    print(f"Number of unique words: {len(unique_words)}")
    print(f"Most common word: '{unique_words[np.argmax(counts)]}' ({np.max(counts)} occurrences)")
    print(f"Data quality - NaN values: EEG={np.isnan(feat).sum()}, Mel={np.isnan(melSpec).sum()}")
    print(f"Data quality - Inf values: EEG={np.isinf(feat).sum()}, Mel={np.isinf(melSpec).sum()}")

    return feat, melSpec, words, feature_names

if __name__ == "__main__":
    # All your participants
    participants = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
                   'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    
    for participant in participants:
        print(f"\n{'='*50}")
        print(f"ANALYZING {participant}")
        print(f"{'='*50}")
        
        # CORRECTED PATH - Check with the right path
        feat_path = f'./data/processed/features/{participant}_feat.npy'
        if os.path.exists(feat_path):
            explore_and_visualize_features(participant)
        else:
            print(f"Features not found for {participant}, skipping...")

