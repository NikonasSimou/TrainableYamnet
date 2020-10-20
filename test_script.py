#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:51:53 2020

@author: nikonas
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from trainable_yamnet import get_trainable_yamnet

'''Simple example of loading data and training a binary YAMNet classifier '''

batch_size = 128
epochs = 10
num_classes = 2
graph = tf.Graph()
with graph.as_default():
    yamnet_binary_model = get_trainable_yamnet(load_weights = True, input_duration = 10/16, num_classes = num_classes)
    Y_train = np.load('/path_to_array_labels.npy')
    X_train = np.load('/path_to_array_of_waveforms.npy')
    Y_train = to_categorical(Y_train,num_classes = 2)
	
    optimizer =  Adam(lr=1e-3)
    yamnet_binary_model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    yamnet_binary_model.summary()

    # preds = yamnet_binary_model.predict(X_train[3:50,:])
    yamnet_binary_model.fit(X_train,Y_train,batch_size = batch_size,epochs = epochs)
