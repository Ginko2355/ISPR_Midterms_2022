import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
from keras_tuner import RandomSearch
import keras_tuner as kt


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
number_of_classes = 10


class HyperCNN(kt.HyperModel):
    def build(self, hp):
        # Moving forward in the layers, the patterns get more complex;
        # hence there are larger combinations of patterns to capture.
        # So I've increased the filter size in subsequent layers to
        # capture as many combinations as possible.

        model = models.Sequential()
        model.add(layers.Conv2D(hp.Int('n_filters_conv2d_1', min_value=16, max_value=64, step=16),
                                kernel_size=hp.Choice('kernel_size', [5, 3]),
                                activation=hp.Choice('f_act_conv_1', ['relu', 'softmax']),
                                input_shape=np.shape(x_train[0])))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))  # the default stride is 2x2

        model.add(layers.Conv2D(hp.Int('n_filters_conv2d_2', min_value=32, max_value=128, step=16),
                                kernel_size=hp.Choice('kernel_size', [5, 3]),
                                activation=hp.Choice('f_act_conv_2', ['relu', 'softmax']),
                                ))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(hp.Int('n_filters_conv2d_3', min_value=64, max_value=256, step=16),
                                kernel_size=hp.Choice('kernel_size', [5, 3]),
                                activation=hp.Choice('f_act_conv_3', ['relu', 'softmax']),
                                ))

        # I can't put another max-pooling here because with 5x5 filters the third conv2d
        # will generate a 1 x 1 x n_filters tensor, which can't be max-pooled with stride=2

        model.add(layers.Flatten())
        model.add(layers.Dense(hp.Int('n_units', min_value=100, max_value=2000, step=100),
                               activation=hp.Choice('f_act_dense', ['relu', 'softmax'])))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", min_value=100, max_value=5000, step=100),
            # Se troppo grande max_value esplode
            **kwargs,
        )


def get_model():
    if len(os.listdir('./best_model')) == 0:
        tuner = RandomSearch(HyperCNN(),
                             objective='val_accuracy',
                             max_trials=50,
                             executions_per_trial=1,
                             directory="tuner_res")

        early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
        model_checkpoint = callbacks.ModelCheckpoint('./best_model', save_best_only=True, mode='auto')
        #Selezione il modello peggiore con "max", provare con "auto" o "min"
        tuner.search(x_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

    return models.load_model('./best_model')
