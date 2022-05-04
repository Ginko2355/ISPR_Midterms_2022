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
from tensorflow.keras import datasets, layers, models, callbacks
from keras_tuner import RandomSearch
import keras_tuner as kt


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
number_of_classes = 10



def build_model(hp):
    # load the dataset

    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('n_filters', min_value=32, max_value=256, step=32),
                            #strides=hp.Choice("strides", [1, 2, 3]),
                            #kernel_size=hp.Choice('kernel_size', [3, 4, 5]),
                            kernel_size=(5, 5),
                            activation='relu',
                            input_shape=np.shape(x_train[0])))

    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(hp.Int('n_filters', min_value=32, max_value=256, step=32),
                            #kernel_size=hp.Choice('kernel_size', [3, 4, 5]),
                            kernel_size=(5, 5),
                            activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('n_units', min_value=100, max_value=1000, step=100), activation='relu'))
    model.add(layers.Dense(number_of_classes))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],)

    return model


tuner = RandomSearch(build_model,
                     objective='val_accuracy',
                     max_trials=1,
                     executions_per_trial=3,
                     directory="tuner_res")

stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=200, validation_split=0.3, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(best_hps)
