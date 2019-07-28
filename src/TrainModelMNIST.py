import keras

def TrainModelMNIST(model, dataset):

    print("Training MNIST model.")
    batch_size = 128
    epochs = 24

    # Compile model
    print("Compiling model.")
    model.compile(loss=keras.losses.categorical_crossentropy,\
                  optimizer=keras.optimizers.Adadelta(),\
                  metrics=['accuracy'])

    # Fit model
    print("Fitting model.")
    model.fit(dataset['x_train'], dataset['y_train'],\
              batch_size=batch_size, epochs=epochs, verbose=1,\
              validation_data=(dataset['x_test'], dataset['y_test']))
    print("MNIST model fit complete.")

    # Save to disk, serialize model to JSON
    model_json = model.to_json()
    with open('cache/model_MNIST.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights('cache/model_MNIST.h5')
    print("MNIST model saved to disk.")

    # Evaluate results
    print("Evaluating model.")
    scores = model.evaluate(dataset['x_test'], dataset['y_test'],\
                            batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

    return



