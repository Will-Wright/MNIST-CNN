import os, sys

sys.path.append(os.getcwd() + "/src")

from GetKerasModelCIFAR import GetKerasModelCIFAR
from TrainKerasModelCIFAR import TrainKerasModelCIFAR
from GetKerasModelMNIST import GetKerasModelMNIST
from TrainKerasModelMNIST import TrainKerasModelMNIST
from PlotKerasModelArchitecture import PlotKerasModelArchitecture

def main(model_name=[], use_cached=True):
    if model_name == 'MNIST':
        run_MNIST = True
        run_CIFAR = False
    elif model_name == 'CIFAR':
        run_MNIST = False
        run_CIFAR = True
    else:
        run_MNIST = True
        run_CIFAR = True

    if not os.path.exists('cache'):
        os.mkdir('cache')

    if run_CIFAR:
        model_Keras_CIFAR, dataset_CIFAR, is_trained_CIFAR \
            = GetKerasModelCIFAR(use_cached)

        if not is_trained_CIFAR:
            PlotKerasModelArchitecture(model_Keras_CIFAR, \
                                       'cache/model_Keras_CIFAR.png')
            TrainKerasModelCIFAR(model_Keras_CIFAR, dataset_CIFAR)

    if run_MNIST:
        model_Keras_MNIST, dataset_MNIST, is_trained_MNIST \
            = GetKerasModelMNIST(use_cached)

        if not is_trained_MNIST:
            PlotKerasModelArchitecture(model_Keras_MNIST, \
                                       'cache/model_Keras_MNIST.png')
            TrainKerasModelMNIST(model_Keras_MNIST, dataset_MNIST)

    return
