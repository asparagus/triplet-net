#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model file."""
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import concatenate
from keras.layers.merge import add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D
from custom_layers import subtract, norm


class TripletNet:

    def __init__(self, shape=(32, 32, 3), dimensions=128):
        self.model = self.build_triplets_model(shape, dimensions)
        self.model.compile(
            loss='mean_squared_error',
            optimizer='sgd',
            metrics=['accuracy']
        )
        self.fit = self.model.fit
        self.fit_generator = self.model.fit_generator
        self.predict = self.model.predict
        self.evaluate = self.model.evaluate
        self.summary = self.model.summary

    def build_triplets_model(self, shape, dimensions):
        net = self.build_embedding(shape, dimensions)

        # Receive 3 inputs
        # Decide which of the two alternatives is closest to the original
        # x  - Original Image
        # x1 - Alternative 1
        # x2 - Alternative 2
        x = Input(shape=shape, name='x')
        x1 = Input(shape=shape, name='x1')
        x2 = Input(shape=shape, name='x2')

        # Get the embedded values
        e = net(x)
        e1 = net(x1)
        e2 = net(x2)

        # Get the differences
        d1 = subtract(e, e1)
        d2 = subtract(e, e2)

        # Normalize the differences
        n1 = norm(d1)
        n2 = norm(d2)

        # Compare
        out = Activation('sigmoid')(subtract(n2, n1))
        return Model(inputs=[x, x1, x2], outputs=out)

    def build_embedding(self, shape, dimensions):
        inp = Input(shape=shape)
        x = inp

        # 3 Conv + MaxPooling + Relu w/ Dropout
        x = self.convolutional_layer(64, kernel_size=5)(x)
        x = self.convolutional_layer(128, kernel_size=3)(x)
        x = self.convolutional_layer(256, kernel_size=3)(x)

        # 1 Final Conv to get into 128 dim embedding
        x = Conv2D(dimensions, kernel_size=2, padding='same')(x)
        x = GlobalMaxPooling2D()(x)

        out = x
        return Model(inputs=inp, outputs=out)

    def convolutional_layer(self, filters, kernel_size):
        def _layer(x):
            x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = Dropout(0.25)(x)
            return x

        return _layer


if __name__ == '__main__':
    t = TripletNet(shape=(32, 32, 3), dimensions=128)
    t.model.summary()
