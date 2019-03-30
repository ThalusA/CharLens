from picamera import PiCamera
from picamera.array import PiRGBArray
import time
def capture_img(img_path='/home/pi/image_camera.jpg', res=(1024,768), vflip=False, hflip=False):
    camera = PiCamera()
    camera.resolution = res
    camera.vflip = vflip
    camera.hflip = hflip
    rawCapture = PiRGBArray(camera, size=res)
    stream = camera.capture_continuous(rawCapture,format="rgb", use_video_port=True)
    frame = None
    on = True
    camera.brightness = 50
    camera.contrast = 60
    time.sleep(0.5)
    print('Demarrage de la previsualisation...')
    camera.start_preview()
    camera.preview_fullscreen = True
    time.sleep(10)
    print('Capture de l'' image...')
    camera.capture(img_path)
    print('Image sauvegardee !'.format(img_path))
    camera.close()    
capture_img()