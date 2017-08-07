import keras
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

from model import TripletNet
from triplets import TripletGenerator


# load the pre-shuffled train and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

input_size = (32, 32, 3)
embedding_dimensions = 128
batch_size = 256
gen = TripletGenerator()
train_stream = gen.flow(x_train, y_train, batch_size=batch_size)
valid_stream = gen.flow(x_valid, y_valid, batch_size=batch_size)

t = TripletNet(shape=input_size, dimensions=embedding_dimensions)
t.summary()

t.model.load_weights('model128big_batch.weights.best.hdf5', by_name=False)
checkpointer = ModelCheckpoint(
    filepath='model128big_batch2.weights.best.hdf5',
    verbose=1, save_best_only=True
)

t.model.fit_generator(
    train_stream, 2500, epochs=30, verbose=1,
    callbacks=[checkpointer],
    validation_data=valid_stream, validation_steps=20
)
# t.model.fit(x_train, y_train, batch_size=64, epochs=50)
