import warnings
warnings.filterwarnings("ignore")

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import os.path


def GetKerasModelMNIST(use_cached):

    # Load fitted model
    if use_cached and os.path.isfile('cache/model_Keras_MNIST.json') and os.path.isfile('cache/model_Keras_MNIST.h5'):
        print("Loading previously trained Keras MNIST model.")
        is_trained = True

        # Load json and create model
        json_file = open('cache/model_Keras_MNIST.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load weights into new model
        model.load_weights("cache/model_Keras_MNIST.h5")

        print("Keras MNIST model loaded from disk.")
        model.summary()

        # Evaluate results
        opt_rms = keras.optimizers.Adadelta()
        print("Compiling model.")
        model.compile(loss=keras.losses.categorical_crossentropy,\
                      optimizer=opt_rms,\
                      metrics=['accuracy'])

        print("Evaluating model.")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        img_rows, img_cols = 28, 28
        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = keras.utils.to_categorical(y_test, num_classes)
        scores = model.evaluate(x_test, y_test,\
                                batch_size=128, verbose=1)
        dataset = []

        print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

    else:
        print("Getting Keras MNIST model.")
        is_trained = False
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        img_rows, img_cols = 28, 28

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        dataset = {'x_train': x_train, 'y_train': y_train,\
                   'x_test': x_test, 'y_test': y_test}

    return model, dataset, is_trained


