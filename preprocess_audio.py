import librosa.feature

from data_loader import load_audio
from utils import calculate_audio_length
from read_config import get_sample_rate
from read_config import get_nfft
from read_config import get_hop_size
from read_config import get_n_mels
from read_config import get_img_size
from pre_processing_data import get_number_of_classes

import numpy as np


def reading_all_audio_files(paths):
    audios = []
    for path in paths:
        audio = load_audio(path)
        audios.append(audio)
    return audios


def cropping_padding(audios, target_length, mode='constant'):
    audio_after_cropping_padding = []
    for audio in audios:
        audio_len = audio.shape[0]
        diff = abs(audio_len - target_length)
        if target_length > audio_len:
            pad1 = np.random.randint(0, diff + 1)
            pad2 = diff - pad1
            audio = np.pad(audio, (pad1, pad2), mode=mode)
        elif target_length < audio_len:
            idx = np.random.randint(0, diff + 1)
            audio = audio[idx: idx + target_length]
        audio_after_cropping_padding.append(audio)
    return audio_after_cropping_padding


def calculate_mel_spectogram(audios):
    spectograms = []
    for audio in audios:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=get_sample_rate(), n_fft=get_nfft(),
                                                  hop_length=get_hop_size(), n_mels=get_n_mels())
        mel_spectrogram_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectograms.append(mel_spectrogram_db)
    return spectograms


def applying_imagenet_preprocessing(spectogram):
    normalized_audios = []
    for specto in spectogram:
        mean = np.mean(specto)
        std = np.std(specto)
        if std == 0:
            spec = specto - mean
        else:
            spec = (specto - mean) / std

        min_val = np.min(spec)
        max_val = np.max(spec)
        if max_val - min_val == 0:
            spec = spec - min_val
        else:
            spec = (spec - min_val) / (max_val - min_val)
        normalized_audios.append(spec)
    return normalized_audios


def process_labels(labels):
    one_hot_targets = []
    for label in labels:
        target = np.array([label])
        target_label = np.zeros(get_number_of_classes(), dtype=np.float32)
        target_label[target] = 1.0
        one_hot_targets.append(target_label)
        print(target_label)
        print(target_label.shape)
    return one_hot_targets


def converting_to_three_channel(normalized_audios):
    converted_audios = []
    for audio in normalized_audios:
        spec = np.expand_dims(audio, axis=-1)
        spec = np.tile(spec, (1, 1, 3))
        spec = np.resize(spec, (*get_img_size(), 3))
        converted_audios.append(spec)
    return converted_audios


def apply_pre_process_audio(paths, labels):
    audios = reading_all_audio_files(paths)
    audios = cropping_padding(audios, target_length=calculate_audio_length())
    audio_spectograms = calculate_mel_spectogram(audios)
    normalized_audios = applying_imagenet_preprocessing(audio_spectograms)
    audios = converting_to_three_channel(normalized_audios)
    labels = process_labels(labels)
    return audios, labels
