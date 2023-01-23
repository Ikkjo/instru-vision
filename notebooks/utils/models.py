import math
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Resizing, Rescaling, Flatten, MaxPool2D, AvgPool2D
from keras.layers import Concatenate, Dropout, RandomFlip, RandomRotation, RandomTranslation, RandomBrightness, RandomContrast, RandomCrop
import tensorflow as tf

RANDOM_FACTOR = .01
BASE_LEARNING_RATE = .0001

def dcnn_model(input_shape=(224, 224, 3), n_classes=6) -> Model:

    height = input_shape[0] 
    width = input_shape[1]

    input = Input(input_shape)

    # Augmentation

    x = image_augmentation_block(input)

    # Image preprocessing

    x = Resizing(height, width)(x)
    x = Rescaling(1/127., offset=-1)(x)

    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = Conv2D(64, 1, activation='relu')(x)
    x = Conv2D(192, 3, padding='same', activation='relu')(x)
    x = MaxPool2D(3, strides=2)(x)

    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D(3, strides=2, padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    x = AvgPool2D(7, strides=1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model

def mobilenetv3_model(input_shape=(224, 224, 3), n_classes=6):
    pass

def inception_block(input, filters):
    t1 = Conv2D(filters[0], 1, activation='relu')(input)

    t2 = Conv2D(filters[1], 1, activation='relu')(input)
    t2 = Conv2D(filters[2], 3, padding='same', activation='relu')(t2)

    t3 = Conv2D(filters[3], 1, activation='relu')(input)
    t3 = Conv2D(filters[4], 5, padding='same', activation='relu')(t3)

    t4 = MaxPool2D(3, 1, padding='same')(input)
    t4 = Conv2D(filters[5], 1, activation='relu')(t4)

    output = Concatenate()([t1, t2, t3, t4])
    return output

def image_augmentation_block(input, output_image_size = (224, 224), random_factor = RANDOM_FACTOR):

    height = output_image_size[0]
    width = output_image_size[1]

    h_crop = math.ceil(height * (1. - random_factor))
    w_crop = math.ceil(width * (1. - random_factor))

    # Image augmentation

    x = RandomFlip("horizontal")(input)
    x = RandomCrop(h_crop, w_crop)(x)
    x = RandomRotation(random_factor)(x)
    x = RandomTranslation(height_factor=random_factor, width_factor=random_factor)(x)
    x = RandomBrightness(random_factor)(x)
    output = RandomContrast(random_factor)(x)

    return output

def compile_model(model: Model) -> Model:
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.optimizers.Adam(BASE_LEARNING_RATE),
              metrics=['accuracy'])

    return model
