import tensorflow as tf
import numpy as np
from scipy import io as spio
import os 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import h5py
from PIL import Image


def load_dataset():
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
    return (x_train, x_test, y_train, y_test)


def normalize_dataset(x_train, x_test, y_train, y_test):
    #Set every variable to float for memory purposes
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    #Applying StandardScaler() to the dataset
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    #Reshape dataset for compability with the first 2D Layer
    x_train = x_train.reshape(x_train.shape[0] ,28, 28, order="A")
    x_test = x_test.reshape(x_test.shape[0] ,28, 28, order="A")
    return (x_train, x_test, y_train, y_test)

def show_graphs(history):
    if isinstance(history, list):
        print("Loss : ", history[0])
        print("Accuracy : ", history[1])
    else : 
        loss_curve = history.history["loss"]
        acc_curve = history.history["accuracy"]
        #Plot loss curve
        plt.plot(loss_curve, color="red", label="Loss Curve")
        plt.title("Loss")
        plt.show()
        #Plot accuracy curve
        plt.plot(acc_curve, color="green", label="Accuracy Curve")
        plt.title("Accuracy")
        plt.show()

def model_layer():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(141, activation="relu"))
    model.add(tf.keras.layers.Dense(47, activation="softmax"))
    return model

if __name__ == "__main__":
    #Load and Normalize dataset
    x_train, x_test, y_train, y_test = load_dataset()
    x_train, x_test, y_train, y_test = normalize_dataset(x_train, x_test, y_train, y_test)
    #Check if there is an already existing model save and ask if you want to load it or create another model
    model_available = list()
    for i, files in enumerate(os.listdir("./models")):
        model_available.append(files)
        print(i, " : ", files[:-3])
    if len(model_available):
        choose = input("Models available, which one you want to load (left blank for creating a new one) ? : ")
    if len(choose):
        model = tf.keras.models.load_model("./models/"+str(model_available[int(choose)]), compile=False)
    elif choose == "":
        model = model_layer()
    #Display basic information about the model and compile it with various function
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    #Ask if you want to train the model or want to test it
    trainBoolean = str(input("Do you want to test the model or train the model (Y: Train Model ; N: Test Model) : ")).upper()
    if trainBoolean == "Y":
        train_step = input("Veuillez renseigner le nombre d'étape d'apprentissage que vous voulez éxécuter : ")
        #Train model
        history = model.fit(x_train, y_train, epochs=int(train_step), use_multiprocessing=True)
        #Save model
        model_name = input("How do you want to call the weight file (left blank for default : 'model_weights') ? : ")
        if model_name == "": model_name = "model_weights"
        tf.keras.models.save_model(model, model_name+'.clw', overwrite=True, include_optimizer=True)
    elif trainBoolean == "N":
        #Test the model
        history = model.evaluate(x_test, y_test, use_multiprocessing=True)
    #Plot accuracy & loss graph
    show_graphs(history)
    




