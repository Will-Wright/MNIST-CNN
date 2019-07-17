# ADD CITATIONS



Description
-----------

This repository contains two different functions, each of which trains a convolutional neural network on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

* The function `TrainCNN` achieves a nearly state-of-the-art accuracy of 99.2%.  This CNN takes about an hour to train on a modern laptop (e.g., my MacBook Air with a 1.7 GHz Intel Core i5 processor and 4 GB of memory).  Import and run `TrainCNN` to generate the following results:

> Epoch 0, last batch accuracy: 100.0000%, valid. accuracy: 98.1800%, valid. best loss: 0.055883  
> Epoch 1, last batch accuracy: 98.0000%, valid. accuracy: 98.8400%, valid. best loss: 0.047431  
> Epoch 2, last batch accuracy: 100.0000%, valid. accuracy: 98.6200%, valid. best loss: 0.040835  
> Epoch 3, last batch accuracy: 100.0000%, valid. accuracy: 98.8600%, valid. best loss: 0.032635  
> Epoch 4, last batch accuracy: 100.0000%, valid. accuracy: 98.7000%, valid. best loss: 0.032635  
> Epoch 5, last batch accuracy: 100.0000%, valid. accuracy: 99.0200%, valid. best loss: 0.032635  
> Epoch 6, last batch accuracy: 100.0000%, valid. accuracy: 99.0400%, valid. best loss: 0.032635  
> Epoch 7, last batch accuracy: 100.0000%, valid. accuracy: 99.0000%, valid. best loss: 0.032635  
> Epoch 8, last batch accuracy: 100.0000%, valid. accuracy: 99.0000%, valid. best loss: 0.032635  
> Epoch 9, last batch accuracy: 100.0000%, valid. accuracy: 98.9400%, valid. best loss: 0.032635  
> Epoch 10, last batch accuracy: 100.0000%, valid. accuracy: 99.1800%, valid. best loss: 0.032635  
> Epoch 11, last batch accuracy: 100.0000%, valid. accuracy: 98.9600%, valid. best loss: 0.032635  
> Epoch 12, last batch accuracy: 100.0000%, valid. accuracy: 98.7400%, valid. best loss: 0.032635  
> Epoch 13, last batch accuracy: 100.0000%, valid. accuracy: 99.1600%, valid. best loss: 0.032635  
> Early stopping!  
> Final accuracy on test set: 0.992

* The function `TrainCNN_quick` also achieves a high accuracy of 98.9%, and only takes about 15 minutes. 




Dependencies
------------

* `tensorflow`
* `numpy`

