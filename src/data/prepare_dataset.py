import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_ljspeech_metadata(dataset_path, output_path, val_split=0.1):
    """
    Prepares the metadata files for LJSpeech dataset for training and validation.
    Assumes .wav files and corresponding transcriptions are available.
    """

    # Read metadata.csv
    metadata_file = os.path.join(dataset_path, "metadata.csv")
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")

    metadata = pd.read_csv(metadata_file, sep='|', header=None, names=['id', 'text', 'normalized_text'])

    # Create new columns
    metadata['wav_path'] = metadata['id'].apply(lambda x: os.path.join(dataset_path, "wavs", f"{x}.wav"))
    metadata = metadata[['wav_path', 'text']]  # Only keep wav path and text

    # Split into train and val
    train_meta, val_meta = train_test_split(metadata, test_size=val_split, random_state=42)

    # Save train and val splits
    train_meta.to_csv(os.path.join(output_path, "metadata_train.csv"), index=False)
    val_meta.to_csv(os.path.join(output_path, "metadata_val.csv"), index=False)

    print(f"Prepared metadata. Train samples: {len(train_meta)}, Val samples: {len(val_meta)}")
