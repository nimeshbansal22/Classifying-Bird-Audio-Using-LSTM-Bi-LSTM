from data_loader import calculate_mfcc
from read_config import get_mfcc
from read_config import get_sample_rate
from read_config import get_fixed_duration
from augmentor import apply_augmentation
from augmentor import augment_mfcc
from read_config import get_filename_with_aug_train
from read_config import get_filename_with_aug_valid

import os
import json
import numpy as np
import librosa
import librosa.feature
import soundfile as sf


def get_shape(data):
    return data.shape


def calculate_audio_length():
    return get_sample_rate() * get_fixed_duration()


def create_mfcc_data(meta_data, n_mfcc=get_mfcc()):
    birds_audio_mfcc = []
    birds_audio_labels = []

    for i, paths in enumerate(meta_data['filepath']):
        for path in paths:
            audio_file_path = os.path.join(path, meta_data.primary_label.iloc[i])
            mfcc = calculate_mfcc(audio_file_path, n_mfcc)
            birds_audio_mfcc.append(mfcc)
            birds_audio_labels.append(meta_data.target_class.iloc[i])
    return birds_audio_mfcc, birds_audio_labels


def save_to_json(data_mfcc, data_labels, filename='mfcc_data'):
    data_mfcc = [mfcc.tolist() if isinstance(mfcc, np.ndarray) else mfcc for mfcc in data_mfcc]
    data_labels = [int(label) if isinstance(label, np.integer) else label for label in data_labels]
    data = {
        "mfcc": data_mfcc,
        "labels": data_labels
    }

    with open("mfcc_data", "w") as f:
        json.dump(data, f)
    print(f"Data saved to {filename}")


def load_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def add_extra_dimension(inputs_train, inputs_validation):
    inputs_train = np.expand_dims(inputs_train, -1)
    inputs_validation = np.expand_dims(inputs_validation, -1)
    return inputs_train, inputs_validation


def process_audio(audio_file, duration=15, sr=32000, n_mfcc=40):
    # Load 15 seconds of audio
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)

    # Check if the audio is shorter than 15 seconds
    if len(y) < duration * sr:
        return None

    # Calculate MFCC for the 15-second audio
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfcc.T


def calculate_mfcc_with_augmentation(audio_file_path, augment=True):
    # Load the audio file with soundfile
    signal, sr = sf.read(audio_file_path, frames=int(get_sample_rate() * get_fixed_duration()))

    # Resample the audio if the sample rate is not 32,000 Hz
    if sr != get_sample_rate():
        signal = librosa.resample(signal, orig_sr=sr, target_sr=get_sample_rate())
        sr = get_sample_rate()

    if augment:
        # Apply augmentations if required
        signal = apply_augmentation(signal, sr)

    # Calculate the target length in samples
    target_length = int(get_fixed_duration() * sr)

    # Crop or pad the signal to ensure it is exactly `target_length` samples long
    if len(signal) < target_length:
        pad_width = target_length - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')

    # Calculate the MFCCs from the audio signal, using the specified number of coefficients
    mfcc = np.mean(librosa.feature.mfcc(y=signal, n_mfcc=get_mfcc(), sr=sr).T, axis=0)

    return mfcc


def save_to_json_for_aug(data_mfcc, data_labels, batch_number, is_train=True, output_dir="output"):
    """Save the processed MFCC data and labels to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    dataset_type = "train" if is_train else "valid"
    filename = os.path.join(output_dir, f"{dataset_type}_mfcc_data_batch_{batch_number}.json")

    # Create a dictionary to store the data
    data_mfcc = [mfcc.tolist() if isinstance(mfcc, np.ndarray) else mfcc for mfcc in data_mfcc]
    data_labels = [int(label) if isinstance(label, np.integer) else label for label in data_labels]

    data = {
        f"{dataset_type}_mfcc": data_mfcc,
        f"{dataset_type}_labels": data_labels
    }

    # Save the data to a JSON file
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Data saved to {filename}")


def process_and_save_in_batches(dataset, batch_size=1000, is_train=True, output_dir="output"):
    total_files = len(dataset)
    for i in range(0, total_files, batch_size):
        batch_mfcc = []
        batch_labels = []
        batch_end = min(i + batch_size, total_files)
        print(f"Processing batch {i // batch_size + 1} / {total_files // batch_size + 1}")

        for j in range(i, batch_end):
            path = dataset.filepath.iloc[j]
            label = dataset.target_class.iloc[j]
            augment = is_train  # Apply augmentation only to training data
            mfcc = calculate_mfcc_with_augmentation(path, get_mfcc(), augment=augment)
            batch_mfcc.append(mfcc)
            batch_labels.append(label)

        save_to_json_for_aug(batch_mfcc, batch_labels, i // batch_size + 1, is_train, output_dir)


def load_json_data_of_aug():
    # Lists to store the arrays
    mfcc_data_list = []
    values_data_list = []

    # Iterate through the file names from batch 1 to batch 23
    for i in range(1, 24):
        # Construct the filename
        file_name = f'train_mfcc_data_batch_{i}.json'
        file_path = os.path.join(get_filename_with_aug_train(), file_name)

        # Check if the file exists before trying to open it
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Load the JSON data
                data = json.load(file)

            # Convert 'train_mfcc' and 'train_values' to NumPy arrays
            mfcc_array = np.asarray(data['train_mfcc'])
            values_array = np.asarray(data['train_labels'])

            # Append the arrays to the respective lists
            mfcc_data_list.append(mfcc_array)
            values_data_list.append(values_array)

            print(f"Successfully processed {file_name}")
        else:
            print(f"{file_name} not found")
    for i in range(1,4):
        file_name = f'valid_mfcc_data_batch_{i}.json'
        file_path = os.path.join(get_filename_with_aug_valid(), file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                # Load the JSON data
                data = json.load(file)

            # Convert 'train_mfcc' and 'train_values' to NumPy arrays
            mfcc_array = np.asarray(data['valid_mfcc'])
            values_array = np.asarray(data['valid_labels'])

            # Append the arrays to the respective lists
            mfcc_data_list.append(mfcc_array)
            values_data_list.append(values_array)

            print(f"Successfully processed {file_name}")
        else:
            print(f"{file_name} not found")
    return mfcc_data_list, values_data_list
