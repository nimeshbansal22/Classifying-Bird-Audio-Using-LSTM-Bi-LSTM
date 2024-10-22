from data_loader import load_data
from pre_processing_data import preprocess_metadata
from read_config import get_base_path
from read_config import is_eda_enabled_for_metadata
from read_config import is_eda_enabled_for_audio
from read_config import is_augmentor_activated
from read_config import get_filename_with_aug_train
from read_config import get_filename_with_aug_valid
from read_config import is_augmentor_activated_for_different_augmentation
from ExploratoryDataAnalysis import training_meta_data
from ExploratoryDataAnalysis import training_audio_data
from pre_processing_data import converting_to_numpy_array
from splitting_the_data import splitting
from utils import add_extra_dimension
from models import basic_lstm_model
from models import basic_bi_lstm
from models import bi_lstm_modified
from models import bi_lstm_for_aug
from models import lstm_model_768
from models import lstm_model_for_different_aug
from models import lstm_model_for_different_aug_1
from models import lstm_model_for_different_aug_2
from models import lstm_model_for_different_aug_3
from models import lstm_model_for_different_aug_4
from models import lstm_model_for_different_aug_5
from plotting_graph import plotting_the_graph_after_training
from utils import process_and_save_in_batches
from utils import load_json_data_of_aug
from utils import process_audio
from preprocess_audio import apply_pre_process_audio
from splitting_the_data import splitting_data
from read_config import get_filename_without_aug
from utils import create_mfcc_data
from utils import save_to_json
from utils import load_from_json
from data_loader import create_paths_labels_datas
import keras
from augmentor import augment_mfcc
import tensorflow as tf

import numpy as np
import imagenet_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    data = load_data(get_base_path())
    if is_eda_enabled_for_metadata():
        training_meta_data.eda(data)
    data = preprocess_metadata(data)
    if is_eda_enabled_for_audio():
        training_audio_data.audio_eda(data)
    data = data.sample(frac=1).reset_index(drop=True)
    audio_mfcc, audio_labels = create_mfcc_data(data)
    save_to_json(audio_mfcc, audio_labels)
    data = load_from_json(get_filename_without_aug())
    inputs, targets = converting_to_numpy_array(data)
    targets_categorical = keras.utils.to_categorical(targets)
    inputs_train, targets_train, inputs_validation, targets_validation = splitting(targets_categorical, inputs)
    inputs_train, inputs_validation = add_extra_dimension(inputs_train, inputs_validation)

    # Creating LSTM model
    lstm_model = basic_lstm_model.create_model_LSTM(inputs_train)
    lstm_model.summary()
    callbacks = basic_lstm_model.callbacks()
    history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                 targets_validation, lstm_model, callbacks)
    plotting_the_graph_after_training(history)

    # Creating Bi directional LSTM
    bi_model = basic_bi_lstm.create_bi_lstm(inputs_train)
    bi_model.summary()
    callbacks = basic_lstm_model.callbacks()
    history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                 targets_validation, bi_model, callbacks)
    plotting_the_graph_after_training(history)

    # creating modified bi directional
    modified_bi_model = bi_lstm_modified.modified_bi_lstm_model(inputs_train)
    modified_bi_model.summary()
    callbacks = basic_lstm_model.callbacks()
    history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                 targets_validation, modified_bi_model, callbacks)
    plotting_the_graph_after_training(history)

    if is_augmentor_activated():
        train_data, valid_data = splitting(data)
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        process_and_save_in_batches(train_data, batch_size=1000, is_train=True,
                                    output_dir=get_filename_with_aug_train())
        process_and_save_in_batches(valid_data, batch_size=1000, is_train=False,
                                    output_dir=get_filename_with_aug_valid())
        mfcc_data_list, values_data_list = load_json_data_of_aug()
        inputs = np.concatenate(mfcc_data_list)
        targets = np.concatenate(values_data_list)
        targets_categorical = keras.utils.to_categorical(targets)
        inputs_train, targets_train, inputs_validation, targets_validation = splitting(targets_categorical, inputs)
        inputs_train, inputs_validation = add_extra_dimension(inputs_train, inputs_validation)
        model = bi_lstm_for_aug.bidirectional_lstm_model_for_augmentation(inputs_train)
        model.summary()
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)


        # model with changes
        model = lstm_model_768.bi_lstm_768(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)

    if is_augmentor_activated_for_different_augmentation():
        data = data.sample(frac=1).reset_index(drop=True)
        data = data.iloc[0:10]
        audio_files = data['filepath']
        audio_labels = data['target_class']
        mfccs = []
        labels = []

        for audio_file, label in tqdm(zip(audio_files, audio_labels), total=len(audio_files)):
            mfcc = process_audio(audio_file)
            if mfcc is not None:
                mfccs.append(mfcc)
                labels.append(label)
            else:
                print(f"Skipping {audio_file} - duration less than 15 seconds")

        X_train, X_val, y_train, y_val = train_test_split(mfccs, labels, test_size=0.1, random_state=42)
        X_train_augmented = []
        y_train_augmented = []

        for mfcc, label in zip(X_train, y_train):
            X_train_augmented.append(mfcc)
            y_train_augmented.append(label)

            # 50% chance of augmentation
            if np.random.rand() < 0.5:
                aug_mfcc = augment_mfcc(mfcc)
                X_train_augmented.append(aug_mfcc)
                y_train_augmented.append(label)

        X_train_augmented = np.array(X_train_augmented)
        y_train_augmented = np.array(y_train_augmented)

        # Process and save training data in batches
        process_and_save_in_batches(X_train_augmented, y_train_augmented, is_train=True, output_dir="output_full_mfcc_with_aug")

        # Save validation data
        process_and_save_in_batches(X_val, y_val, is_train=False, output_dir="output_full_mfcc_with_aug_labels")

        model = lstm_model_for_different_aug.create_model_LSTM(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)



        #next model
        model = lstm_model_for_different_aug_1.create_model_LSTM(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)



        #next model
        model = lstm_model_for_different_aug_2.create_model_LSTM(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)


        # next model
        model = lstm_model_for_different_aug_3.create_simple_LSTM_with_logit_shifting(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)


        #next model
        model = lstm_model_for_different_aug_4.create_model_LSTM(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)


        #next model
        model = lstm_model_for_different_aug_5.create_model_LSTM(inputs_train)
        callbacks = basic_lstm_model.callbacks()
        history = basic_lstm_model.fitting_the_model(inputs_train, targets_train, inputs_validation,
                                                     targets_validation, model, callbacks)
        plotting_the_graph_after_training(history)

    #trying to implement efficientnet
    train_data, valid_data = splitting_data(data)
    train_paths = train_data.filepath.values
    train_labels = train_data.target_class.values
    train_audios, train_labels = apply_pre_process_audio(train_paths, train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_audios, train_labels))
    print(tf.shape(train_dataset))

    train_data = create_paths_labels_datas(train_data, True, True)
    validation_data = create_paths_labels_datas(valid_data, False, False)
    model = imagenet_model.modelling()
    checkpoint_weights = imagenet_model.pre_trained_model_checkpoint()
    history = imagenet_model.model_history(model, train_data, validation_data, checkpoint_weights)
    best_epoch = np.argmax(history.history["val_auc"])
    best_score = history.history["val_auc"][best_epoch]
    print('>>> Best AUC: ', best_score)
    print('>>> Best Epoch: ', best_epoch + 1)


if __name__ == "__main__":
    main()
