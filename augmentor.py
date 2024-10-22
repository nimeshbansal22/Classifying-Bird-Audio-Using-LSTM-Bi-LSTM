import keras_cv
import random
import librosa
import tensorflow as tf
import numpy as np


def build_augmenter():
    augmenters = [
        keras_cv.layers.MixUp(alpha=0.4),
        keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0),
                                     width_factor=(0.06, 0.12)),  # time-masking
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1),
                                     width_factor=(1.0, 1.0)),  # freq-masking
    ]

    def augment(img, label):
        data = {"images": img, "labels": label}
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.35:
                data = augmenter(data, training=True)
        return data["images"], data["labels"]

    return augment


def apply_augmentation(signal, sr):
    # Randomly apply time stretching
    if random.random() < 0.5:
        rate = random.uniform(0.8, 1.2)
        signal = librosa.effects.time_stretch(signal, rate=rate)

    # Randomly apply pitch shifting
    if random.random() < 0.5:
        steps = random.randint(-2, 2)
        signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)

    # Randomly add background noise
    if random.random() < 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(signal)
        signal = signal + noise_amp * np.random.normal(size=signal.shape[0])

    return signal


def augment_mfcc(mfcc):
    # Time warping
    time_warp_factor = np.random.uniform(0.9, 1.1)
    mfcc_time_warped = librosa.effects.time_stretch(mfcc.T, rate=time_warp_factor).T

    # Frequency masking
    freq_mask = np.random.randint(0, mfcc.shape[1] // 5)
    freq_start = np.random.randint(0, mfcc.shape[1] - freq_mask)
    mfcc_freq_masked = mfcc_time_warped.copy()
    mfcc_freq_masked[:, freq_start:freq_start + freq_mask] = 0

    return mfcc_freq_masked
