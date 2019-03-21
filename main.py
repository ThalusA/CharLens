import tensorflow as tf
import numpy as np
from scipy import io as spio
import os 
import matplotlib.pyplot as plt

emnist = spio.loadmat("emnist-balanced.mat")

# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

train_labels = y_train
test_labels = y_test

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 1 ,28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0], 1 ,28, 28, order="A")

y_train = tf.keras.utils.to_categorical(y_train, 47)
y_test = tf.keras.utils.to_categorical(y_test, 47)



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(81, activation="relu"))
model.add(tf.keras.layers.Dense(47, activation="softmax"))

model.summary()

model.compile(
	loss="sparse_categorical_crossentropy",
	optimizer="sgd",
	metrics=["accuracy"]
)

history = model.fit(x_train, y_train, epochs=10)
loss_curve = history.history["loss"]
acc_curve = history.history["acc"]

plt.plot(loss_curve)
plt.title("Loss")
plt.show()

plt.plot(acc_curve)
plt.title("Accuracy")
plt.show()
