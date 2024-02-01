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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model = keras.models.load_model("ConvNet.h5") # mmwave_CNN.h5
model.summary()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

left_swipe_train = pd.concat(map(pd.read_csv, ["data/left_swipe/left_swipe_3.csv"]), ignore_index=True)
right_swipe_train = pd.concat(map(pd.read_csv, ["data/right_swipe/right_swipe_4.csv"]), ignore_index=True)
up_swipe_train = pd.concat(map(pd.read_csv, ["data/up_swipe/up_swipe_3.csv"]), ignore_index=True)
down_swipe_train = pd.concat(map(pd.read_csv, ["data/down_swipe/down_swipe_3.csv"]), ignore_index=True)
forward_palm_train = pd.concat(map(pd.read_csv, ["data/forward_palm/forward_palm_3.csv"]), ignore_index=True)

x_train = []
my_idx = ["x", "y", "z", "Doppler"]
idx_size = len(my_idx)
sample_size = 3

encoded_labels = ["Left Swipe", "Right Swipe", "Up Swipe", "Down Swipe", "Forward Palm"]
encoded_labels = np.array(encoded_labels)


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

print("X train value before resize: ", x_train)
# x_train = np.asarray(x_train).astype("float32")
x_train = np.asarray(x_train).reshape(-1, sample_size, idx_size).astype("float32")


left_swipe_y = np.full(int(len(left_swipe_train)/sample_size), 0)
right_swipe_y = np.full(int(len(right_swipe_train)/sample_size), 1)
up_swipe_y = np.full(int(len(up_swipe_train)/sample_size), 2)
down_swipe_y = np.full(int(len(down_swipe_train)/sample_size), 3)
forward_palm_y = np.full(int(len(forward_palm_train)/sample_size), 4)

y_train = np.concatenate((left_swipe_y, right_swipe_y, up_swipe_y, down_swipe_y, forward_palm_y))
print("X train value before split: ", x_train.shape)
print("Y train value: ", y_train.shape)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1, random_state = 42)

#y_pre = model.predict(x_test)
#prediction = encoded_labels[tf.argmax(model.predict(x_test), axis=1)]
prediction = tf.argmax(model.predict(x_train), axis=1)
print("The predicted y values are: ", prediction)
print("X train value: ", x_train.shape)
print("Y train value: ", y_train.shape)
cf_matrix = confusion_matrix(y_train, prediction)

model.evaluate(x_test, y_test, batch_size=8, verbose=2)

#print(cf_matrix)

cf_graph = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt = '.2%', cmap='Blues')

cf_graph.set_title('Confusion Matrix Showing Predicted Gesture Vs the Actual Gesture \n\n')
cf_graph.set_xlabel('\nPredicted Values')
cf_graph.set_ylabel('\nActual Values ')

# Set Matrix Label Names
cf_graph.xaxis.set_ticklabels(["Left Swipe", "Right Swipe", "Up Swipe", "Down Swipe", "Forward Palm"])
cf_graph.yaxis.set_ticklabels(["Left Swipe", "Right Swipe", "Up Swipe", "Down Swipe", "Forward Palm"])

# Display the Confusion Matrix.
plt.show()