import keras
from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


def TrainModelCIFAR(model, dataset):

    print("Training CIFAR-10 model.")
    batch_size = 64
    epochs = 125
    opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

    # Compile model
    print("Compiling model.")
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms,\
                  metrics=['accuracy'])

    # Fit model
    print("Fitting model.")
    model.fit_generator(dataset['datagen'].flow(dataset['x_train'],\
                                                dataset['y_train'],\
                                                batch_size=batch_size),\
                        steps_per_epoch=dataset['x_train'].shape[0] // batch_size,\
                        epochs=epochs,\
                        verbose=1,validation_data=(dataset['x_test'],\
                        dataset['y_test']),\
                        callbacks=[LearningRateScheduler(lr_schedule)])
    print("CIFAR-10 model fit complete.")

    # Save to disk, serialize model to JSON
    model_json = model.to_json()
    with open('cache/model_CIFAR.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights('cache/model_CIFAR.h5')
    print("CIFAR-10 model saved to disk.")

    # Evaluate results
    print("Evaluating model.")
    scores = model.evaluate(dataset['x_test'], dataset['y_test'],\
                            batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))

    return



