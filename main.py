from PIL import Image
import numpy
import tensorflow as tf
import os

prediction_dict = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "A",
    "B",
    "D",
    "E",
    "F",
    "G",
    "H",
    "N",
    "Q",
    "R",
    "T"
]

if __name__ == "__main__":
    model_available = list()
    for i, files in enumerate(os.listdir("./models")):
        model_available.append(files)
        print(i, " : ", files[:-4])
    if len(model_available):
        choose = input("Models available, which one you want to load ? : ")
    if len(choose):
        model = tf.keras.models.load_model("./models/"+str(model_available[int(choose)])+".clw", compile=False)
    else:
        print("Please don't left it blank.")
        quit()
    if(input("Do you want to use the Rasberry's camera or load 'input.png' ( Left blank for using camera )")):
        from camera import capture_img
        capture_img("./input.png")
    im = Image.open("input.png")
    np_im = numpy.array(im.convert("L"))
    np_im = numpy.expand_dims(np_im, axis=0)
    prediction = model.predict(np_im)
    print("The prediction says it's a : ", prediction_dict[numpy.where(prediction == numpy.amax(prediction))[1][0]])

