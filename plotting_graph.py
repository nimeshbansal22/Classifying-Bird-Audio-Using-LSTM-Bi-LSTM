import pandas as pd
import matplotlib.pyplot as plt


def plotting_the_graph_after_training(history):
    history_frame = pd.DataFrame(history.history)
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Plot loss and val_loss
    axs[0].plot(history_frame['loss'], label='Training Loss')
    axs[0].plot(history_frame['val_loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot categorical_accuracy and val_categorical_accuracy
    axs[1].plot(history_frame['categorical_accuracy'], label='Training Accuracy')
    axs[1].plot(history_frame['val_categorical_accuracy'], label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()