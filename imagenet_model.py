import keras
import keras_cv

from pre_processing_data import get_number_of_classes
from read_config import get_epoch
from read_config import get_batch_size


class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, lr_start, lr_max, lr_min, total_epochs):
        super().__init__()
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.total_epochs // 3:
            lr = self.lr_start + (self.lr_max - self.lr_start) * (epoch / (self.total_epochs // 3))
        else:
            lr = self.lr_max + (self.lr_min - self.lr_max) * (
                        (epoch - self.total_epochs // 3) / (2 * self.total_epochs // 3))
        self.model.optimizer.learning_rate.assign(lr)
        print(f"\nEpoch {epoch + 1}: Learning rate is {lr:.2e}")


def pre_trained_model_checkpoint():
    checkpoint_best_model_weights = keras.callbacks.ModelCheckpoint("best_model.weights.h5",
                                                                    monitor='val_auc',
                                                                    save_best_only=True,
                                                                    save_weights_only=True,
                                                                    mode='max')
    return checkpoint_best_model_weights


def model_history(model, train_meta_data, validation_meta_data, model_checkpoint):
    batch_size = get_batch_size()
    total_epochs = get_epoch()

    lr_start, lr_max, lr_min = 5e-5, 8e-6 * batch_size, 1e-5
    lr_scheduler = CustomLearningRateScheduler(lr_start, lr_max, lr_min, total_epochs)

    history = model.fit(
        train_meta_data,
        validation_data=validation_meta_data,
        epochs=total_epochs,
        callbacks=[
            model_checkpoint,
            lr_scheduler
        ],
        verbose=1
    )
    return history


def modelling():
    print("Starting model creation...")
    input_data = keras.layers.Input(shape=(None, None, 3))
    print("Input layer created")

    # Pretrained backbone
    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset('efficientnetv2_l')
    print("Backbone created")

    classifier = keras_cv.models.ImageClassifier(
        backbone=backbone,
        num_classes=get_number_of_classes(),
        name="classifier"
    )
    print("Classifier created")

    out = classifier(input_data)
    print("Classifier output created")

    # Build model
    model = keras.models.Model(inputs=input_data, outputs=out)
    print("Model built")

    # Compile model with optimizer, loss and metrics
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
            metrics=[keras.metrics.AUC(name='auc')]
        )
        print("Model compiled successfully")
    except Exception as e:
        print(f"Error during model compilation: {str(e)}")
        return None

    print("Model summary:")
    model.summary()
    return model
