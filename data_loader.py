import pandas as pd
import soundfile as sf
import os
import librosa
import numpy as np
import librosa.feature

from read_config import get_sample_rate
from read_config import get_train_metadata_filename
from read_config import get_batch_size
from read_config import is_augmentor_activated
from Building_dataset_for_model import build_dataset


def load_data(base_path):
    meta_data = pd.read_csv(os.path.join(base_path, get_train_metadata_filename()))
    return meta_data


def load_audio(filepath):
    return librosa.load(path=filepath, sr=get_sample_rate(), mono=True, duration=15)


def calculate_mfcc(audio_filepath, mfcc):
    audio, sr = load_audio(audio_filepath)
    mfcc = np.mean(librosa.feature.mfcc(y=audio, n_mfcc=mfcc, sr=sr).T, axis = 0)
    return mfcc


def create_paths_labels_datas(meta_data, shuffle, augment):
    paths = meta_data.filepath.values
    targets = meta_data.target_class.values
    if shuffle & augment:
        data = build_dataset(
            paths, targets, batch_size=get_batch_size(),
            shuffle=True, augment=is_augmentor_activated())
    else:
        data = build_dataset(
            paths, targets, batch_size=get_batch_size(),
            shuffle=False, augment=False)
    return data
