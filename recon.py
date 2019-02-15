import cv2
import numpy as np

def main():
    confidence_ratio = 0.90
    image = cv2.imread('input.jpg')
    temp_image = np.zeros(image.shape, np.uint8)
    temp_image.fill(255)
    detector = cv2.text_TextDetectorCNN.create("model.prototxt", "icdar13.caffemodel")
    Bbox, confidence = detector.detect(image)
    filtered = list()
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
    image = temp_image
    print("Detected : ", len(filtered), " strings")
    for box in filtered:
        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2],box[1]+box[3]), (0, 255, 0), 1)
        cropped = cv2.cvtColor(image[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])], cv2.COLOR_BGR2GRAY)
        cropped = cv2.inRange(cropped, 100, 245)
        height, width = cropped.shape
        temp_image = np.zeros((height+20, width+20), np.uint8)
        temp_image[10:(height+10), 10:(width+10)] = cropped
        contours, hierarchy = cv2.findContours(temp_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
        cv2.drawContours(temp_image, contours, -1, (255, 255, 255))
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h in range((height//3)*2, height):
                cv2.rectangle(image, (box[0]+x-10, box[1]+y-10), (box[0]-10+x+w, box[1]-10+y+h), (255,0,0), 1)
    cv2.imwrite('output.jpg', image)

main()
