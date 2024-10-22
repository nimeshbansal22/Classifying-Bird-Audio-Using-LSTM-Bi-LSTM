from os import listdir
from os.path import join
from read_config import get_train_audiodata_filepath

import numpy as np


def get_number_of_classes():
    return len(listdir(get_train_audiodata_filepath()))


def convert_from_classes_to_labels():
    classes_name = sorted(listdir(get_train_audiodata_filepath()))
    class_to_label = {class_name: index for index, class_name in enumerate(classes_name)}
    return class_to_label


def modifying_filename_column(meta_data):
    xc_id_list = []
    for name in meta_data.filename:
        xc_id = name.split('/')[1]
        xc_id_list.append(xc_id)
    meta_data['filename'] = xc_id_list
    return meta_data


def generating_target_column(meta_data, class_to_label):
    target_class_list = []
    for name in meta_data.primary_label:
        if name in class_to_label:
            target_class_list.append(class_to_label[name])
    meta_data['target_class'] = target_class_list
    return meta_data


def generate_audio_file_paths(meta_data):
    filepaths = []
    for name in meta_data.filename:
        filepath = join(get_train_audiodata_filepath(), name)
        filepaths.append(filepath)
    meta_data['filepath'] = filepaths
    return meta_data


def preprocess_metadata(meta_data):
    meta_data = generate_audio_file_paths(meta_data)
    meta_data = modifying_filename_column(meta_data)
    class_to_label = convert_from_classes_to_labels()
    meta_data = generating_target_column(meta_data, class_to_label)
    return meta_data


def converting_to_numpy_array(meta_data):
    audio_mfcc = meta_data['mfcc']
    audio_targets = meta_data['labels']
    inputs = np.asarray(audio_mfcc)
    targets = np.array(audio_targets)
    return inputs, targets

