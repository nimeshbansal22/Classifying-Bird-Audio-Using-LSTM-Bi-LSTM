from decoder import build_decoder
from augmentor import build_augmenter
import tensorflow as tf

from read_config import get_seed
from utils import calculate_audio_length


def build_dataset(paths, labels=None, batch_size=32,
                  decode_fn=True, augment_fn=True, cache=True,
                  augment=False, shuffle=2048):
    if decode_fn:
        decode_fn = build_decoder(labels is not None, dim=calculate_audio_length())

    if augment_fn:
        augment_fn = build_augmenter()
    slices = (paths,) if labels is None else (paths, labels)
    print(slices)
    print(type(slices))
    meta_data = tf.data.Dataset.from_tensor_slices(slices)
    meta_data = meta_data.map(decode_fn)
    meta_data = meta_data.cache() if cache else meta_data
    if shuffle:
        opt = tf.data.Options()
        meta_data = meta_data.shuffle(shuffle, seed=get_seed())
        opt.experimental_deterministic = False
        meta_data = meta_data.with_options(opt)
    meta_data = meta_data.batch(batch_size, drop_remainder=True)
    meta_data = meta_data.map(augment_fn) if augment else meta_data
    meta_data = meta_data.prefetch(4)
    return meta_data
