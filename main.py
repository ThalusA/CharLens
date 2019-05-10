from PIL import Image
import numpy
import tensorflow as tf
import os
import train_eval
import camera as camera_program
import recon

prediction_dict = ["0", "1", "2",  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", 
    "V", "W", "X", "Y", "Z", "A", "B", "D", "E", "F", "G", "H", "N", "Q", "R", "T"] # Assigne une variable dictionaire des divers representation alphanumérique ...
    # ... des resultat numérique de prédiction

if __name__ == "__main__":
    model_available = list()
    if(input("Voulez-vous rentre dans une phase d'apprentissage/testage (Y/n) n par défault : ") == "Y"):
        train_eval.main()
        quit()
    for i, files in enumerate(os.listdir("./models")):
        model_available.append(files)
        print(i, " : ", files[:-4])
    if len(model_available):
        choose = input("Modèles disponibles, lequels voulez-vous charger ? : ")
    if len(choose):
        model = tf.keras.models.load_model("./models/"+str(model_available[int(choose)]), compile=False)
    else:
        print("Ne laissez pas ceci vide.")
        quit()
    # Demande si le chargement d'un modèle d'IA est nécessaire puis en selectionne un choisit
    if(not(input("Voulez-vous utiliser la caméra du Raspberry ou charger le fichier image 'input.png' (Laisser vide pour utiliser la caméra)"))):
        camera_program.capture_img()
    # Demande de quel manière l'image à mettre dans l'IA est obtenue (Par le Raspberry ou une image stockée)
    char_list = recon.main() 
    if (not(len(char_list))): 
        print("Aucunes chaînes de caractères n'a été détecté")
        quit()
    for im in char_list:
        np_im = numpy.array(im.convert("L"))
        np_im = numpy.expand_dims(np_im, axis=0)
        # Charge l'image sous un format lisible par l'IA
        prediction = model.predict(np_im) # Prédit par l'IA le caractère montré.
        im.show()
        print("La prédiction dit que c'est un : ", prediction_dict[numpy.where(prediction == numpy.amax(prediction))[1][0]]) # Affiche la prédiction sous forme alphanumérique

