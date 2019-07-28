Description
-----------

This repository contains implementations of two convolutional neural networks (CNNs), 
one for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) 
and one for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).



These two CNNs are based on chapter 13 in [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781491962282/) and the author's [accompanying software](https://github.com/ageron/handson-ml).

Note that TensorFlow versions >= 2.0 will throw deprecated software warnings.

* The function `` achieves a nearly state-of-the-art accuracy of 99.2%.  This CNN takes about 1.5 hours to train on a modern laptop (e.g., my MacBook Air with a 1.7 GHz Intel Core i5 processor and 4 GB of memory).  Import and run `TrainCNN` to generate the following results:

* The function `TrainCNN_quick` also achieves a high accuracy of 98.9%, and only takes about 30 minutes. 


Dependencies
------------

* `tensorflow`
* `numpy`

