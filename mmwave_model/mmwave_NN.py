import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import TensorBoard
from keras.layers import Dense, InputLayer, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

left_swipe_train = pd.concat(map(pd.read_csv, ["data/left_swipe/left_swipe_1.csv", "data/left_swipe/left_swipe_2.csv"]), ignore_index=True)
right_swipe_train = pd.concat(map(pd.read_csv, ["data/right_swipe/right_swipe_2.csv", "data/right_swipe/right_swipe_3.csv"]), ignore_index=True)
up_swipe_train = pd.concat(map(pd.read_csv, ["data/up_swipe/up_swipe_1.csv", "data/up_swipe/up_swipe_2.csv"]), ignore_index=True)
down_swipe_train = pd.concat(map(pd.read_csv, ["data/down_swipe/down_swipe_1.csv", "data/down_swipe/down_swipe_2.csv"]), ignore_index=True)
forward_palm_train = pd.concat(map(pd.read_csv, ["data/forward_palm/forward_palm_1.csv", "data/forward_palm/forward_palm_2.csv"]), ignore_index=True)

x_train = []
my_idx = ["x", "y", "z", "Doppler"]
idx_size = len(my_idx)
sample_size = 3

# left_swipe_train
for i in range(int(len(left_swipe_train)/sample_size)):
    x_train.append(left_swipe_train[sample_size*i:sample_size*i+sample_size][my_idx].values.tolist())

# right_swipe_train
for i in range(int(len(right_swipe_train)/sample_size)):
    x_train.append(right_swipe_train[sample_size*i:sample_size*i+sample_size][my_idx].values.tolist())

# up_swipe_train
for i in range(int(len(up_swipe_train)/sample_size)):
    x_train.append(up_swipe_train[sample_size*i:sample_size*i+sample_size][my_idx].values.tolist())

# down_swipe_train
for i in range(int(len(down_swipe_train)/sample_size)):
    x_train.append(down_swipe_train[sample_size*i:sample_size*i+sample_size][my_idx].values.tolist())

# forward_palm_train
for i in range(int(len(forward_palm_train)/sample_size)):
    x_train.append(forward_palm_train[sample_size*i:sample_size*i+sample_size][my_idx].values.tolist())

# x_train = np.asarray(x_train).astype("float32")
x_train = np.asarray(x_train).reshape(-1, sample_size, idx_size).astype("float32")


left_swipe_y = np.full(int(len(left_swipe_train)/sample_size), 0)
right_swipe_y = np.full(int(len(right_swipe_train)/sample_size), 1)
up_swipe_y = np.full(int(len(up_swipe_train)/sample_size), 2)
down_swipe_y = np.full(int(len(down_swipe_train)/sample_size), 3)
forward_palm_y = np.full(int(len(forward_palm_train)/sample_size), 4)

y_train = np.concatenate((left_swipe_y, right_swipe_y, up_swipe_y, down_swipe_y, forward_palm_y))

x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42)


model = keras.Sequential()
model.add(Conv1D(12, 3, padding="same",activation="relu", input_shape=(sample_size, idx_size)))
model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(5, activation="softmax"))

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics = ["accuracy"],
)

callbacks = [keras.callbacks.TensorBoard(log_dir='logs',
                                         histogram_freq=1,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='epoch',
                                         profile_batch=2,
                                         embeddings_freq=1)]

model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=2, shuffle=True, validation_data=(x_test, y_test), callbacks = callbacks)
print("test set: ")
model.evaluate(x_test, y_test, batch_size=8, verbose=2)
model.summary()
print('Saving model: ')
model.save('mmwave_CNN.h5')
