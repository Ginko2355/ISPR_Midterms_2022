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

import model_selection
from matplotlib import pyplot as plt
from PIL import Image
from cleverhans.tf2.attacks import fast_gradient_method as fgm
from tensorflow.keras import callbacks
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

obj_class ={0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
            }

def get_image(i):
    return x_test[i], y_test[i]


def print_probabilities(pred_array):
    for i in range(len(pred_array)):
        print(obj_class[i] + " with " + format(pred_array[i], '.2f') + " probability.")


def main():
    model = model_selection.get_model()
    early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    model_checkpoint = callbacks.ModelCheckpoint('./best_model', save_best_only=True, mode='max')
    model.summary()
    model.fit(x=x_train, y=y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

    model = model.fit(batch_size=2600)
    model_input_and_output = tf.keras.Model(model.input, model.layers[-1].output)

    image, label = get_image(np.random.randint(x_test.shape[0]))
    image = np.reshape(image, (32, 32, 3))
    # plt.title(obj_class[label[0]])
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    epsilon = 0.6
    adv_image = fgm.fast_gradient_method(model_input_and_output, np.reshape(image, (1, 32, 32, 3)), epsilon, np.inf, targeted=False)
    adv_image_label_pred = model.predict(adv_image)
    image_label_pred = model.predict(np.reshape(image, (1, 32, 32, 3)))

    f, axarr = plt.subplots(1, 2)
    axarr[0].set_title("Recognized as: " + obj_class[np.argmax(image_label_pred)])
    axarr[0].imshow(image)
    axarr[0].axis('off')

    axarr[1].set_title("Recognized as: " + obj_class[np.argmax(adv_image_label_pred)])
    axarr[1].imshow(np.reshape(adv_image, (32, 32, 3)).astype('uint8'))
    axarr[1].axis('off')
    f.tight_layout()
    plt.show()

    print("The Actual label of the image is " + obj_class[label[0]])
    print("\nThe probabilities for the image without adversarial attack is:")
    print_probabilities(image_label_pred[0])
    print("\nThe probabilities for the image with adversarial attack is:")
    print_probabilities(adv_image_label_pred[0])

    print("Model accuracy : " + str(measure_accuracy(model)))
    # plt.title(obj_class[np.argmax(adv_image_label_pred)])
    # plt.imshow(np.reshape(adv_image, (32, 32, 3)).astype('uint8'))
    # plt.axis('off')
    # plt.show()

def measure_accuracy(model):
    accuracy = 0.
    prob_true = 1/10000
    for i in range(len(x_test)):
        prediction = model.predict(np.reshape(x_test[i], (1, 32, 32, 3)))[0]
        if np.argmax(prediction) == y_test[i]:
            accuracy += prob_true

        print(i, "/ 10000")

    return accuracy

if __name__ == '__main__':
    main()
