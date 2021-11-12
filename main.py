from model import *
from data import *
import tensorflow as tf
from keras.models import *
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene = trainGenerator(32,'concentrated/maskfits/maskpngs/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model.fit_generator(myGene,steps_per_epoch=10,epochs=5)

testGene = testGenerator("/home/kupa/kupa/unetfiles/unet-Sydil/3C76/images1/")
results = model.predict_generator(testGene,81,verbose=1)
saveResult("/home/kupa/kupa/unetfiles/unet-Sydil/3C76/predict/",results)
print('prediction masks:', results.shape)
