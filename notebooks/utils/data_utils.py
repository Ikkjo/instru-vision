import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from file_utils import get_class_labels

tf.get_logger().setLevel('ERROR')

DATASET_PATH = "../data/images/"
AUTOTUNE = tf.data.AUTOTUNE

def display_training_results(history, model_name, big_figure=True):
    # Create a figure with two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 11), layout="constrained")

    if big_figure:
        fig.set_size_inches(8, 11)
        fig.set_dpi(300)

    # Plot the accuracy in the first subplot
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot the loss in the second subplot
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.suptitle("Results for model: " + model_name)
    plt.show()

def compare_and_display_model_results(model_histories: dict, big_figure=True):

    # Determine the number of rows and columns for the subplots based on the number of history objects
    n_plots = len(model_histories)
    n_rows = int(n_plots/2) + (n_plots % 2)
    n_cols = 2

    # Create a figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 11*n_rows), layout="constrained")
    axs = axs.ravel()

    if big_figure:
        fig.set_size_inches(8, 11)
        fig.set_dpi(300)

    for i, (model_name, history) in enumerate(model_histories):

        # Set the title of the subplot

        # axs[i].set_title("Model {}:".format(i+1) + model_name)

        # Plot the accuracy in the first column
        axs[i].plot(history.history['accuracy'])
        axs[i].plot(history.history['val_accuracy'])
        axs[i].set_title('Model accuracy')
        axs[i].set_ylabel('Accuracy')
        axs[i].set_xlabel('Epoch')
        axs[i].legend(['Train', 'Validation'], loc='upper left')

        # Plot the loss in the second column
        axs[i+1].plot(history.history['loss'])
        axs[i+1].plot(history.history['val_loss'])
        axs[i+1].set_title('Model loss')
        axs[i+1].set_ylabel('Loss')
        axs[i+1].set_xlabel('Epoch')
        axs[i+1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()


def display_images(images):
    # Determine the number of rows and columns for the subplots based on the number of images
    n_plots = len(images)
    n_cols = 8
    n_rows = int(n_plots/n_cols) + (n_plots % n_cols)

    # Create a figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10), layout="constrained")
    axs = axs.ravel()

    for i, img in enumerate(images):
        # Display the image in the subplot
        axs[i].imshow(img)
        axs[i].set_title("Image {}".format(i+1))
        axs[i].axis("off")

    plt.show()

def get_datasets():
    training_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(DATASET_PATH,
                                                      labels="inferred",
                                                      label_mode='categorical',
                                                      class_names=get_class_labels().keys(),
                                                      color_mode='rgb',
                                                      batch_size=32,
                                                      image_size=(224, 224),
                                                      shuffle=True,
                                                      seed=123,
                                                      validation_split=0.2,
                                                      subset="both")

    training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return training_dataset, validation_dataset
