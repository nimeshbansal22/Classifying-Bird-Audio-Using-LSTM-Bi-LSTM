from pre_processing_data import get_number_of_classes
from read_config import get_hop_size
from read_config import get_frame_size
from read_config import get_img_size
from read_config import get_sample_rate

import keras
import librosa
import tensorflow as tf
import soundfile as sf
from tempfile import NamedTemporaryFile


def build_decoder(with_labels=True, dim=1024):
    def get_audio(filepath):
        file_bytes = tf.io.read_file(filepath)
        with NamedTemporaryFile(suffix=".ogg") as tmpfile:
            tmpfile.write(file_bytes.numpy())
            tmpfile.flush()
            audio, sample_rate = sf.read(tmpfile.name, dtype='float32')
            if sample_rate != 32000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=32000)
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio_tensor
        # audio, sample_rate = librosa.load(filepath, sr=get_sample_rate(), mono=True)
        # audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        # return audio_tensor

    def crop_or_pad(audio, target_len, pad_mode="constant"):
        audio_len = tf.shape(audio)[0]
        diff_len = abs(
            target_len - audio_len
        )  # find difference between target and audio length
        if audio_len < target_len:  # do padding if audio length is shorter
            pad1 = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            pad2 = diff_len - pad1
            audio = tf.pad(audio, paddings=[[pad1, pad2]], mode=pad_mode)
        elif audio_len > target_len:  # do cropping if audio length is larger
            idx = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            audio = audio[idx: (idx + target_len)]
        return tf.reshape(audio, [target_len])

    def apply_preproc(spec):
        # Standardize
        mean = tf.math.reduce_mean(spec)
        std = tf.math.reduce_std(spec)
        spec = tf.where(tf.math.equal(std, 0), spec - mean, (spec - mean) / std)

        # Normalize using Min-Max
        min_val = tf.math.reduce_min(spec)
        max_val = tf.math.reduce_max(spec)
        spec = tf.where(
            tf.math.equal(max_val - min_val, 0),
            spec - min_val,
            (spec - min_val) / (max_val - min_val),
        )
        return spec

    def get_target(target):
        target = tf.reshape(target, [1])
        target = tf.cast(tf.one_hot(target, get_number_of_classes()), tf.float32)
        target = tf.reshape(target, [get_number_of_classes()])
        return target

    def decode(path):
        # Load audio file
        audio = get_audio(path)
        # Crop or pad audio to keep a fixed length
        audio = crop_or_pad(audio, dim)
        # Audio to Spectrogram
        spec = keras.layers.MelSpectrogram(
            num_mel_bins=get_img_size()[0],
            fft_length=get_frame_size(),
            sequence_stride=get_hop_size(),
            sampling_rate=get_sample_rate(),
        )(audio)
        # Apply normalization and standardization
        spec = apply_preproc(spec)
        # Spectrogram to 3 channel image (for imagenet)
        spec = tf.tile(spec[..., None], [1, 1, 3])
        spec = tf.reshape(spec, [*get_img_size(), 3])
        return spec

    def decode_with_labels(path, label):
        label = get_target(label)
        return decode(path), label

    return decode_with_labels if with_labels else decode
