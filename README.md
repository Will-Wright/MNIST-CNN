Description
-----------

This repository contains implementations of two convolutional neural networks (CNNs) for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using TensorFlow.
These two CNNs are based on chapter 13 in [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/) and the author's [accompanying software](https://github.com/ageron/handson-ml).

Note that TensorFlow versions >= 2.0 will throw a few deprecated software warnings.

* The function `TrainCNN` achieves a nearly state-of-the-art accuracy of 99.2%.  This CNN takes about 1.5 hours to train on a modern laptop (e.g., my MacBook Air with a 1.7 GHz Intel Core i5 processor and 4 GB of memory).  Import and run `TrainCNN` to generate the following results:

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

* The function `TrainCNN_quick` also achieves a high accuracy of 98.9%, and only takes about 30 minutes. 

> Epoch 0 Last batch accuracy: 0.97 Test accuracy: 0.9767  
> Epoch 1 Last batch accuracy: 0.99 Test accuracy: 0.9829  
> Epoch 2 Last batch accuracy: 0.99 Test accuracy: 0.9832  
> Epoch 3 Last batch accuracy: 0.97 Test accuracy: 0.9872  
> Epoch 4 Last batch accuracy: 1.0 Test accuracy: 0.9844  
> Epoch 5 Last batch accuracy: 1.0 Test accuracy: 0.9856  
> Epoch 6 Last batch accuracy: 0.99 Test accuracy: 0.986  
> Epoch 7 Last batch accuracy: 1.0 Test accuracy: 0.9887  
> Epoch 8 Last batch accuracy: 1.0 Test accuracy: 0.9866  
> Epoch 9 Last batch accuracy: 1.0 Test accuracy: 0.9878  


Dependencies
------------

* `tensorflow`
* `numpy`

