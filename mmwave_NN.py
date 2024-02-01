import pandas as pd
import numpy as np
import os
from tensorflow import keras
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Removes GPU tensorflow errors

# Extract the gesture data from the CSV files
left_swipe_train = pd.concat(map(pd.read_csv, ["data/left_swipe/left_swipe_1.csv", "data/left_swipe/left_swipe_2.csv"]), ignore_index=True)
right_swipe_train = pd.concat(map(pd.read_csv, ["data/right_swipe/right_swipe_2.csv", "data/right_swipe/right_swipe_3.csv"]), ignore_index=True)
up_swipe_train = pd.concat(map(pd.read_csv, ["data/up_swipe/up_swipe_1.csv", "data/up_swipe/up_swipe_2.csv"]), ignore_index=True)
down_swipe_train = pd.concat(map(pd.read_csv, ["data/down_swipe/down_swipe_1.csv", "data/down_swipe/down_swipe_2.csv"]), ignore_index=True)
forward_palm_train = pd.concat(map(pd.read_csv, ["data/forward_palm/forward_palm_1.csv", "data/forward_palm/forward_palm_2.csv"]), ignore_index=True)

x_train = [] # Empty array for the train data to be stored in
my_idx = ["x", "y", "z", "Doppler"] # List of the columns of data to be taken from the CSV files
idx_size = len(my_idx)
sample_size = 3 # Amount of samples taken, also sets the time sample windows of the 1D ConvNet

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

 # Reshape the array to match the needed input to the 1D convNet
x_train = np.asarray(x_train).reshape(-1, sample_size, idx_size).astype("float32")

# Create the labels for the various gestures
left_swipe_y = np.full(int(len(left_swipe_train)/sample_size), 0)
right_swipe_y = np.full(int(len(right_swipe_train)/sample_size), 1)
up_swipe_y = np.full(int(len(up_swipe_train)/sample_size), 2)
down_swipe_y = np.full(int(len(down_swipe_train)/sample_size), 3)
forward_palm_y = np.full(int(len(forward_palm_train)/sample_size), 4)

# Combine the labels into a single y_train array
y_train = np.concatenate((left_swipe_y, right_swipe_y, up_swipe_y, down_swipe_y, forward_palm_y))

#Split the data into train and validation split.
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42)

# The CNN architecture
model = keras.Sequential()
model.add(Conv1D(12, 3, padding="same",activation="relu", input_shape=(sample_size, idx_size)))
model.add(Dense(16, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(5, activation="softmax")) #Final output layer, number of neurons relates to the amount of classes

#Set the models learning rate and metrics
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001), # 0.0001 found to work the best
    metrics = ["accuracy"],
)

# Set up tensorboard in order to obtain graphs of the training results
callbacks = [keras.callbacks.TensorBoard(log_dir='logs',
                                         histogram_freq=1,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='epoch',
                                         profile_batch=2,
                                         embeddings_freq=1)]

# Train the model on the train data over 100 epochs, 
model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=2, shuffle=True, validation_data=(x_val, y_val), callbacks = callbacks)
model.save('ConvNet.h5') #Save model
model.summary() #Display the models layer information
