from PIL import Image
import numpy
import tensorflow as tf

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
    im = Image.open("image.png")
    np_im = numpy.array(im.convert("L"))
    np_im = numpy.expand_dims(np_im, axis=0)
    model = tf.keras.models.load_model("model_weights", compile=False)
    prediction = model.predict(np_im)
    print("The prediction says it's a : ", prediction_dict[numpy.where(prediction == numpy.amax(prediction))[1][0]])

