import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('input.png', 0)
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU, img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x,y), (x + w, y + h), 255, 1)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    cv2.imwrite('output.png', img)
