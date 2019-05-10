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
    # Charge le set de données
    x_train = emnist["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)
    # Charge les labels du set de données
    y_train = emnist["dataset"][0][0][0][0][0][1]
    # Charge le set de données de test
    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)
    # Charge les labels du set de données de test
    y_test = emnist["dataset"][0][0][1][0][0][1]
    return (x_train, x_test, y_train, y_test)

def normalize_dataset(x_train, x_test, y_train, y_test):
    # Assigne toutes les variables au type "float" pour compatibilité des interventions suivantes
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    # Applique la fonction StandardScaler() au deux set de données
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    # Remodeler les sets de données pour facilité la compabilité avec la couche d'entrée du réseau neuronal de l'IA
    x_train = x_train.reshape(x_train.shape[0] ,28, 28, order="A")
    x_test = x_test.reshape(x_test.shape[0] ,28, 28, order="A")
    return (x_train, x_test, y_train, y_test)

def show_graphs(history):
    if isinstance(history, list):
        print("Perte : ", history[0])
        print("Précision : ", history[1])
    else : 
        loss_curve = history.history["loss"]
        acc_curve = history.history["accuracy"]
        # Fait le graphique de la courbe de perte
        plt.plot(loss_curve, color="red", label="Courbe de Perte")
        plt.title("Perte")
        plt.show()
        # Fait le graphique de la courbe de précision
        plt.plot(acc_curve, color="green", label="Courbe de Précision")
        plt.title("Précision")
        plt.show()

def model_layer():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
    model.add(tf.keras.layers.Dense(256, kernel_size=5, activation="relu"))
    model.add(tf.keras.layers.Dense(141, activation="relu"))
    model.add(tf.keras.layers.Dense(47, activation="softmax"))
    # Création des différentes couches du réseaux neuronales de l'IA
    return model

def main():
    # Charge et normalise les différents set de données
    x_train, x_test, y_train, y_test = load_dataset()
    x_train, x_test, y_train, y_test = normalize_dataset(x_train, x_test, y_train, y_test)
    # Verifier s'il y a déjà un modèle existant puis demande si on veux le charger ou créer un autre modèle
    model_available = list()
    for i, files in enumerate(os.listdir("./models")):
        model_available.append(files)
        print(i, " : ", files[:-4])
    if len(model_available):
        choose = input("Modèle disponible, lequels d'entre eux vouler vous charger (laisser vide pour créer un nouveau modèle) ? : ")
    if len(choose):
        model = tf.keras.models.load_model("./models/"+str(model_available[int(choose)]), compile=False)
    elif choose == "":
        model = model_layer()
    # Affiche les divers information à propos du modèles et le compile avec divers paramètres d'apprentissage
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # Demande s'il faut faire apprendre le modèle ou le tester
    trainBoolean = str(input("Voulez-vous tester le model ou l'entrainer ? (Y: Entrainer le modèle ; N: Tester le modèle) : ")).upper()
    if trainBoolean == "Y":
        train_step = input("Veuillez renseigner le nombre d'étape d'apprentissage que vous voulez éxécuter : ")
        # Entraine le modèle
        history = model.fit(x_train, y_train, epochs=int(train_step), use_multiprocessing=True) 
        # Sauvegarde le modèle
        model_name = input("Comment voulez-vous nommer le fichier du modèle (laisser vide pour mettre par défault : 'model_weights') ? : ")
        if not len(model_name): 
            model_name = "model_weights"
        tf.keras.models.save_model(model, "./models/"+model_name+'.clw', overwrite=True, include_optimizer=True)
    elif trainBoolean == "N":
        # Test le modèle
        history = model.evaluate(x_test, y_test, use_multiprocessing=True)
    # Affiche sous forme de graphique un résumé de la session d'apprentissage
    show_graphs(history)




