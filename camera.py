from picamera import PiCamera
import time

def capture_img():
    camera = PiCamera()
    camera.resolution = (2592, 1944)
    print('Demarrage de la previsualisation...')
    camera.start_preview()
    time.sleep(10)
    print("Capture de l'image...")
    camera.capture('./input.png')
    camera.stop_preview()
    print('Image sauvegardee dans : ./input.png')
    camera.close()
