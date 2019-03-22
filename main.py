import tensorflow as tf
import numpy as np
from scipy import io as spio
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import h5py

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

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_train = x_train.astype(float)
x_test = x_test.astype(float)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train = x_train.reshape(x_train.shape[0] ,28, 28, order="A")
x_test = x_test.reshape(x_test.shape[0] ,28, 28, order="A")

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

tf.keras.models.save_model(
    model,
    'model_weights',
    overwrite=True,
    include_optimizer=True
)


loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

plt.plot(loss_curve)
plt.title("Loss")
plt.show()

plt.plot(acc_curve)
plt.title("Accuracy")
plt.show()
