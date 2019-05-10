import cv2 # Importe l'api OpenCV
import numpy as np # Importe le module "numpy"

def main():
    confidence_ratio = 0.8  #Assigne une valeur de confiance au résultat du detecteur de chaines de caractères prenant donc que les prédictions qui ont plus de 90% de chance d'être juste
    image = cv2.imread('input.png') #Charge l'image à analyser
    temp_image = np.zeros(image.shape, np.uint8) #Créer une image temporaire où stocker toutes les chaines de caractères detecter
    temp_image.fill(255) #Remplit cette image temporaire d'un fond blanc
    detector = cv2.text_TextDetectorCNN.create("model.prototxt", "icdar13.caffemodel") #Charge l'IA reconnaisseuse d'emplacement de chaines de caractères préentrainé avec le set de ...
    #... données "icdar13" tout en chargeant la configuration du model depuis "model.prototxt"
    Bbox, confidence = detector.detect(image) # Fait tourner l'IA sur l'image à analyser pour obtenir ses prédictions sur les différentes chaînes de caractère detecté
    filtered = list() # Créer un tableau qui servira à filtrer les délimitations chaînes de caractères qui se chevaucher pour les rassembler en une seul délimitation
    for i in range(len(Bbox)):
        if confidence[i] > confidence_ratio:
            temp_image[Bbox[i][1]:(Bbox[i][1]+Bbox[i][3]), Bbox[i][0]:(Bbox[i][0]+Bbox[i][2])] = image[Bbox[i][1]:(Bbox[i][1]+Bbox[i][3]), Bbox[i][0]:(Bbox[i][0]+Bbox[i][2])]
            overwrited = False
            for n in range(len(filtered)):
                rect_points = [(Bbox[i][0], Bbox[i][1]),
                               (Bbox[i][0]+Bbox[i][2], Bbox[i][1]), 
                               (Bbox[i][0], Bbox[i][1]+Bbox[i][3]), 
                               (Bbox[i][0]+Bbox[i][2], Bbox[i][1]+Bbox[i][3])]
                for point in rect_points:
                    if point[0] in range(filtered[n][0], filtered[n][0]+filtered[n][2]) and point[1] in range(filtered[n][1], filtered[n][1]+filtered[n][3]):
                        overwrited = True
                if overwrited == True:
                    if Bbox[i][0] < filtered[n][0]:
                        filtered[n][2] += (filtered[n][0]-Bbox[i][0])
                        filtered[n][0] = Bbox[i][0]
                    if Bbox[i][1] < filtered[n][1]:
                        filtered[n][3] += (filtered[n][1]-Bbox[i][1])
                        filtered[n][1] = Bbox[i][1]
                    if (Bbox[i][0]+Bbox[i][2]) > (filtered[n][0]+filtered[n][2]):
                        filtered[n][2] = (Bbox[i][0]+Bbox[i][2])-filtered[n][0]
                    if (Bbox[i][1]+Bbox[i][3]) > (filtered[n][1]+filtered[n][3]):
                        filtered[n][3] = (Bbox[i][1]+Bbox[i][3])-filtered[n][1]
            if overwrited == False: filtered.append(Bbox[i])
    # Filtre les délimitations chaînes de caractères qui se chevaucher pour les rassembler en une seul délimitation
    image = temp_image # Créer une image identique servant de visualisation
    print("Détecté : ", len(filtered), " chaines de caractères") # Affiche le nombre de chaînes de caractères detecté
    char_list = list()
    for box in filtered: # Traite chaque chaines de caractères afin d'extraire les différents caractères les composant.
        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2],box[1]+box[3]), (0, 255, 0), 1)
        cropped = cv2.cvtColor(image[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])], cv2.COLOR_BGR2GRAY)
        cropped = cv2.inRange(cropped, 100, 245)
        # Reduit le bruit de l'image 
        height, width = cropped.shape
        temp_image = np.zeros((height+20, width+20), np.uint8)
        temp_image[10:(height+10), 10:(width+10)] = cropped
        # Ajoute une marge de visualisation autour de la délimitation pour être sûr d'englober la chaînes de caractères entièrement
        contours, hierarchy = cv2.findContours(temp_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        cv2.drawContours(temp_image, contours, -1, (255, 255, 255))
        # Délimite les contours des caractères contenant la chaines de caractères étudié
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h in range((height//3)*2, height):
                cropped_char = image[box[1]+y-10:box[1]+y-10+h, box[0]+x-10:box[0]-10+x+w]
                resized_char = None
                height_chr, width_chr = cropped_char.shape
                cv2.resize(cropped_char,resized_char,28/width_chr, 28/height_chr, interpolation = cv2.INTER_CUBIC)
                char_list.append(resized_char)
                cv2.rectangle(image, (box[0]+x-10, box[1]+y-10), (box[0]-10+x+w, box[1]-10+y+h), (255,0,0), 1)
        # Dessine les contours des différences caractères en rajoutant une légèrement marge pour un meilleur englobement et visualisation 
    return(char_list)
    # Sauvegarde l'image traité et analysé dans le fichier "output.jpg"
