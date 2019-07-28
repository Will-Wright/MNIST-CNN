import os, sys

sys.path.append(os.getcwd() + "/src")

from GetModelCIFAR import GetModelCIFAR
from TrainModelCIFAR import TrainModelCIFAR
from GetModelMNIST import GetModelMNIST
from TrainModelMNIST import TrainModelMNIST
from PlotModelArchitecture import PlotModelArchitecture

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
        model_CIFAR, dataset_CIFAR, is_trained_CIFAR = GetModelCIFAR(use_cached)

        if not is_trained_CIFAR:
            PlotModelArchitecture(model_CIFAR, 'cache/model_CIFAR.png')
            TrainModelCIFAR(model_CIFAR, dataset_CIFAR)

    if run_MNIST:
        model_MNIST, dataset_MNIST, is_trained_MNIST = GetModelMNIST(use_cached)

        if not is_trained_MNIST:
            PlotModelArchitecture(model_MNIST, 'cache/model_MNIST.png')
            TrainModelMNIST(model_MNIST, dataset_MNIST)

    return
