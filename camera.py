from picamera import PiCamera
import time
def capture_img(img_path='/home/pi/image_camera.jpg', res=(1024,768), vflip=False, hflip=False):
    camera = PiCamera(resolution=res, vflip=vflip, hflip=hflip, constrat=60)
    print('Demarrage de la previsualisation...')
    camera.start_preview()
    time.sleep(10)
    print("Capture de l'image...")
    camera.capture(img_path)
    camera.stop_preview()
    print('Image sauvegardee dans : {}'.format(img_path))
    camera.close()