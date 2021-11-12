# Semantic Segmentation of Astronomical Radio Images: A Computer Vision Approach 

The model was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

Two datasets were used, ATLBS survey and MeerKAT. The pre-processed images have a sixe of 256 x 256 pixels each.

### Model

This deep neural network is implemented with Keras.

### Training

The model is trained for 5 epochs.

After 5 epochs, calculated accuracy is about 0.9980.

Loss function used is a binary crossentropy.

---

## How to use

### Dependencies

Required libraries:

* Tensorflow
* Keras >= 1.0

This code is compatible with Python versions 2.7-3.5.

### Run main.py for segmentation
### Run centroid.py for Detection

