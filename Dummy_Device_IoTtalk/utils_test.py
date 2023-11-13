import os
from Detic.detect_custom_obj import create_detector, detect_object

PREDICTOR = None
def build():
    global PREDICTOR
    os.chdir('./Detic')
    PREDICTOR = create_detector('cpu')
    os.chdir('..')
    print(os.getcwd())
    
def detect():
    _, num = detect_object(PREDICTOR,
                  '/home/samel/workplace/ntou/iottalk/dummy_detic/desk.jpg',
                  classes=['webcam', 'headphone'],
                  view=True)


if __name__=="__main__":
    build()
    detect()
