import cv2
import numpy as np

if __name__ == "__main__":
    large = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
    height, width, channels = large.shape
    rgb = cv2.pyrDown(large)
    mod_height, mod_width, mod_channels = rgb.shape
    height, width  = height/mod_height, width/mod_width
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    i = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, i, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(large, (int(x*width), int(y*height)), (int((x+w-1)*width), int((y+h-1)*height)), (0, 255, 0), 2)
        i += 1
    cv2.imwrite('output.png', large)