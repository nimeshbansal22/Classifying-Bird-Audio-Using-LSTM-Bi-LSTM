import yaml
from os.path import join


def get_config():
    cfg = yaml.full_load(open("./config.yml", 'r'))
    train_config = cfg['TRAIN']
    return train_config


def get_saved_data_config():
    cfg = yaml.full_load(open("./config.yml", 'r'))
    saved_data_config = cfg['SAVED_DATA']
    return saved_data_config


def get_base_path():
    cfg = get_config()
    return cfg['BASE_DATA_PATH']


def get_sample_rate():
    cfg = get_config()
    return cfg['SAMPLE_RATE']


def get_hop_size():
    cfg = get_config()
    return cfg['HOP_SIZE']


def get_frame_size():
    cfg = get_config()
    return cfg['FRAME_SIZE']


def get_n_mels():
    cfg = get_config()
    return cfg['N_MELS']


def get_train_metadata_filename():
    cfg = get_config()
    return cfg['TRAIN_METADATA_FILENAME']


def get_train_audiodata_filepath():
    cfg = get_config()
    return join(get_base_path(), cfg['TRAIN_AUDIODATA_FILENAME'])


def get_fixed_duration():
    cfg = get_config()
    return cfg['FIXED_AUDIO_DURATION']


def get_fmin():
    cfg = get_config()
    return cfg['F_MIN']


def get_fmax():
    cfg = get_config()
    return cfg['F_MAX']


def get_img_size():
    cfg = get_config()
    return cfg['IMG_SIZE']


def is_eda_enabled_for_metadata():
    cfg = yaml.full_load(open("./config.yml", 'r'))
    eda_config = cfg['EDA']
    return eda_config['IS_ACTIVATED_FOR_METADATA']


def is_eda_enabled_for_audio():
    cfg = yaml.full_load(open("./config.yml", 'r'))
    eda_config = cfg['EDA']
    return eda_config['IS_ACTIVATED_FOR_AUDIO']


def get_seed():
    cfg = get_config()
    return cfg['SEED']


def is_decoder_activated():
    cfg = get_config()
    return cfg['IS_DECODER_ACTIVATED']


def is_augmentor_activated():
    cfg = get_config()
    return cfg['IS_AUGMENTATION_ACTIVATED']


def get_batch_size():
    cfg = get_config()
    return cfg['BATCH_SIZE']


def get_epoch():
    cfg = get_config()
    return cfg['EPOCHS']


def get_nfft():
    cfg = get_config()
    return cfg['N_FFT']


def get_mfcc():
    cfg = get_config()
    return cfg['N_MFCC']


def get_filename_without_aug():
    cfg = get_saved_data_config()
    return cfg['OUTPUT_FILENAME_WITHOUT_AUG']


def get_filename_with_aug():
    cfg = get_saved_data_config()
    return cfg['OUTPUT_FILENAME_WITH_AUG']


def get_filename_with_aug_train():
    cfg = get_filename_with_aug()
    return cfg['TRAIN']


def get_filename_with_aug_valid():
    cfg = get_filename_with_aug()
    return cfg['VALID']


def is_augmentor_activated_for_different_augmentation():
    cfg = get_config()
    return cfg['IS_AUGMENTATION_ACTIVATED_FOR_DIFFERENT_AUGMENTATION']


def get_filename_with_diff_aug():
    cfg = get_saved_data_config()
    return cfg['OUTPUT_FILENAME_WITH_DIFFERENT_AUG']


def get_filename_with_diff_aug_train():
    cfg = get_filename_with_diff_aug()
    return cfg['TRAIN']


def get_filename_with_diff_aug_valid():
    cfg = get_filename_with_diff_aug()
    return cfg['VALID']
