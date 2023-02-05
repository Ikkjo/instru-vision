import os
import sys
module_path = os.path.abspath(os.path.join('utils'))
if module_path not in sys.path:
    sys.path.append(module_path)

import tensorflow as tf
import utils.data_utils as data_utils
import time
from utils.models import dcnn_model, mobilenetv3_model, compile_model, image_augmentation_block
from keras import callbacks


tf.get_logger().setLevel('ERROR')
BASE_LEARNING_RATE = .0001
EPOCHS = 150
INSTRU_VISION_CNN = f"instru-vision-cnn-{int(time.time())}"
MOBILENETV3_FINE_TUNED = f"mobilenetv3-fine-tuned-{int(time.time())}"
EARLY_STOPPING = callbacks.EarlyStopping(monitor = "val_loss",
                            mode = "min",
                            min_delta = .01,
                            patience = 15,
                            restore_best_weights = True)



DCNN_CALLBACKS = [
    callbacks.CSVLogger(
        "../models/instru-vision-cnn/csv/training_history.csv",
        ",",
        append=False),
    callbacks.TensorBoard(log_dir=f"../models/logs/{INSTRU_VISION_CNN}"),
    callbacks.ModelCheckpoint(
    filepath="../models/instru-vision-cnn/saved_model/",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    initial_value_threshold=0.9),
    EARLY_STOPPING
    ]

PRETRAINED_CALLBACKS = [
    callbacks.CSVLogger(
        "../models/mobilenetv3/csv/training_history.csv",
        ",",
        append=False),
    callbacks.TensorBoard(log_dir=f"../models/logs/{MOBILENETV3_FINE_TUNED}"),
    callbacks.ModelCheckpoint(
    filepath="../models/mobilenetv3/saved_model/",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    initial_value_threshold=0.6)
]

training_dataset, validation_dataset = data_utils.get_datasets()

pretrained_model = mobilenetv3_model()
# pretrained_model.summary()
compile_model(pretrained_model)
pretrained_history = pretrained_model.fit(training_dataset.map(lambda images, labels: (image_augmentation_block(images), labels)),
                                          validation_data=validation_dataset,
                                          verbose=1,
                                          epochs=EPOCHS,
                                          callbacks=PRETRAINED_CALLBACKS)

pretrained_model.save_weights(f"../models/mobilenetv3/best-weights/mobilenetv3_weights.hdf5")

dcnn_model = dcnn_model()
# dcnn_model.summary()
compile_model(dcnn_model)
dcnn_history = dcnn_model.fit(training_dataset,
                              validation_data=validation_dataset,
                              verbose=1,
                              epochs=EPOCHS,
                              callbacks=DCNN_CALLBACKS)
dcnn_model.save_weights(f"../models/instru-vision-cnn/best-weights/dcnn_weights.hdf5")
