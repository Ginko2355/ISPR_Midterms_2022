# Implement your own convolutional network, deciding how many layers,
# the type of layers and how they are interleaved, the type of pooling,
# the use of residual connections, etc. Discuss why you made each choice
# a provide performance results of your CNN on CIFAR-10.

# Now that your network is trained, you might try an adversarial attack to it.
# Try the simple Fast Gradient Sign method, generating one (or more) adversarial
# examples starting from one (or more) CIFAR-10 test images. It is up to you to
# decide if you want to implement the attack on your own or use one of the available
# libraries (e.g. foolbox,  CleverHans, ...). Display the original image, the adversarial
# noise and the final adversarial example.
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
from keras_tuner import RandomSearch
import keras_tuner as kt

#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
number_of_classes = 10


class HyperCNN(kt.HyperModel):
    def build(self, hp):
        # Moving forward in the layers, the patterns get more complex;
        # hence there are larger combinations of patterns to capture.
        # So I've increased the filter size in subsequent layers to
        # capture as many combinations as possible.

        model = models.Sequential()
        model.add(layers.Conv2D(hp.Int('n_filters_conv2d_1', min_value=32, max_value=64, step=32),
                                kernel_size=hp.Choice('kernel_size', [5, 3]),
                                activation='relu',
                                input_shape=np.shape(x_train[0])))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))  # the default stride is 2x2

        model.add(layers.Conv2D(hp.Int('n_filters_conv2d_2', min_value=64, max_value=128, step=64),
                                kernel_size=hp.Choice('kernel_size', [5, 3]),
                                activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(hp.Int('n_filters_conv2d_3', min_value=128, max_value=256, step=128),
                                kernel_size=hp.Choice('kernel_size', [5, 3]),
                                activation='relu'))

        # I can't put another max-pooling here because with 5x5 filters the third conv2d
        # will generate a 1 x 1 x n_filters tensor, which can't be max-pooled with stride=2

        model.add(layers.Flatten())
        model.add(layers.Dense(hp.Int('n_units', min_value=100, max_value=2000, step=100), activation='softmax'))
        model.add(layers.Dense(number_of_classes, activation='softmax'))
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Int("batch_size", min_value=32, max_value=128, step=32),
            **kwargs,
        )


tuner = RandomSearch(HyperCNN(),
                     objective='val_accuracy',
                     max_trials=50,
                     executions_per_trial=2,
                     directory="tuner_res")

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=200, validation_split=0.3, callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps)

# CNN Side fatta, da vedere come salvare ed utilizzare i best_hps.
# Da fare: Adversarial attack using foolbox or CleverHans, print the images, the noises
#          and show how the CNN can classify better the images without the noise.
