from model import *
from data import *
import tensorflow as tf
from keras.models import *
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

model = unet()
model.fit_generator(trainGenerator(32,'concentrated/maskfits/maskpngs/train','image','label',
                       rotation_range=0.2,
                     rotation_range=1.50,
                     rotation_range=0.90,
                     width_shift_range=0.05,
                     width_shift_range=0.2,
                     width_shift_range=0.4,
                    height_shift_range=0.05,
                    height_shift_range=0.2,
                    height_shift_range=0.4,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest,save_to_dir = None),steps_per_epoch=10,epochs=5)
saveResult("/home/kupa/kupa/unetfiles/unet-Sydil/3C76/predict/",model.predict_generator(testGenerator("/home/kupa/kupa/unetfiles/unet-Sydil/3C76/images1/"),81,verbose=1))

