from sklearn.model_selection import train_test_split


def splitting_data(meta_data):
    train_meta_data, valid_meta_data = train_test_split(meta_data, test_size=0.1, random_state=42)
    print(f"Num Train: {len(train_meta_data)} | Num Valid: {len(valid_meta_data)}")
    return train_meta_data, valid_meta_data


def splitting(targets_categorical, inputs):
    nos = inputs.shape[0]
    training_samples = int(nos * 0.9)
    validation_samples = int(nos * 0.1)

    inputs_train = inputs[:training_samples]
    targets_train = targets_categorical[:training_samples]
    inputs_validation = inputs[training_samples:training_samples + validation_samples]
    targets_validation = targets_categorical[training_samples:training_samples + validation_samples]
    return inputs_train, targets_train, inputs_validation, targets_validation
