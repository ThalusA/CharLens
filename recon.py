import cv2
import numpy as np

def main():
    image = cv2.imread('input.jpg')
    result = list()
    detector = cv2.text_TextDetectorCNN.create("model.prototxt", "icdar13.caffemodel")
    Bbox, confidence = detector.detect(image)
    for i in range(len(Bbox)):
        if confidence[i] > 0.90:
            result.append(Bbox[i])
            cv2.rectangle(image, (Bbox[i][0], Bbox[i][1]), (Bbox[i][0]+Bbox[i][2],Bbox[i][1]+Bbox[i][3]), (0, 255, 0), 2)
    cv2.imwrite('output.jpg', image)

main()
