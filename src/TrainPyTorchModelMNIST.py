import warnings
warnings.filterwarnings("ignore")


import numpy as np
import os.path


def GetPyTorchModelMNIST(use_cached):

    # Load fitted model
    if use_cached and os.path.isfile('cache/model_CIFAR.json') and os.path.isfile('cache/model_CIFAR.h5'):

        print("Loading previously trained CIFAR-10 model.")
        is_trained = True

        # Load json and create model
        json_file = open('cache/model_CIFAR.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # Load weights into new model
        model.load_weights("cache/model_CIFAR.h5")

        print("CIFAR-10 model loaded from disk.")
        model.summary()

        # Evaluate results
        opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
        print("Compiling model.")
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms,\
                      metrics=['accuracy'])

        print("Evaluating model.")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_test = x_test.astype('float32')
        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))
        x_test = (x_test-mean)/(std+1e-7)
        num_classes = 10
        y_test = np_utils.to_categorical(y_test,num_classes)
        scores = model.evaluate(x_test, y_test,\
                                batch_size=128, verbose=1)

        dataset = []

        print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

    else:
        print("Getting CIFAR-10 model.")
        is_trained = False
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #z-score
        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))
        x_train = (x_train-mean)/(std+1e-7)
        x_test = (x_test-mean)/(std+1e-7)

        num_classes = 10
        y_train = np_utils.to_categorical(y_train,num_classes)
        y_test = np_utils.to_categorical(y_test,num_classes)

        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same',\
                  kernel_regularizer=regularizers.l2(weight_decay),\
                  input_shape=x_train.shape[1:]))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same',\
                  kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), padding='same',\
                  kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same',\
                  kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3,3), padding='same',\
                  kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same',\
                  kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        # Augment data
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            )
        datagen.fit(x_train)

        dataset = {'x_train': x_train, 'y_train': y_train,\
                   'x_test': x_test, 'y_test': y_test,\
                   'datagen': datagen}

    return model, dataset, is_trained





