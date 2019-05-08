from PIL import Image
import numpy
import tensorflow as tf
import os

prediction_dict = ["0", "1", "2",  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
    "V", "W", "X", "Y", "Z", "A", "B", "D", "E", "F", "G", "H", "N", "Q", "R", "T"] # Assigne une variable dictionaire des divers representation alphanumérique ...
    # ... des resultat numérique de prédiction

if __name__ == "__main__":
    model_available = list()
    for i, files in enumerate(os.listdir("./models")):
        model_available.append(files)
        print(i, " : ", files[:-4])
    if len(model_available):
        choose = input("Modèles disponibles, lequels voulez-vous charger ? : ")
    if len(choose):
        model = tf.keras.models.load_model("./models/"+str(model_available[int(choose)])+".clw", compile=False)
    else:
        print("Ne laissez pas ceci vide.")
        quit()
    # Demande si le chargement d'un modèle d'IA est nécessaire puis en selectionne un choisit
    if(input("Voulez-vous utiliser la caméra du Raspberry ou charger le fichier image 'input.png' ( Laisser vide pour utiliser la caméra )")):
        from camera import capture_img
        capture_img("./output.png")
    # Demande de quel manière l'image à mettre dans l'IA est obtenue (Par le Raspberry ou une image stockée)
    im = Image.open("output.png")
    np_im = numpy.array(im.convert("L"))
    np_im = numpy.expand_dims(np_im, axis=0)
    # Charge l'image sous un format lisible par l'IA
    prediction = model.predict(np_im) # Prédit par l'IA le caractère montré.
    print("La prédiction dit que c'est un : ", prediction_dict[numpy.where(prediction == numpy.amax(prediction))[1][0]]) # Affiche la prédiction sous forme alphanumérique

