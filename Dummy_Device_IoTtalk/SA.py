import random 
import os
from Detic.detect_custom_obj import create_detector, detect_object

PREDICTOR = None
def build():
    global PREDICTOR
    os.chdir('./Detic')
    PREDICTOR = create_detector('cpu')
    os.chdir('..')
    print(os.getcwd())
build()

ServerID = 3
ServerURL = 'https://{}.iottalk.tw'.format(ServerID) #For example: 'https://iottalk.tw'
MQTT_broker = '{}.iottalk.tw'.format(ServerID) # MQTT Broker address, for example:  'iottalk.tw' or None = no MQTT support
MQTT_port = 5566
MQTT_encryption = True
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'Obj_Count'
IDF_list = ['Object_Detect']
ODF_list = ['Dummy_Control']
device_id = None #if None, device_id = MAC address
device_name = None
exec_interval = 1  # IDF/ODF interval

def on_register(r):
    print('Server: {}\nDevice name: {}\nRegister successfully.'.format(r['server'], r['d_name']))

def Dummy_Sensor():
    return random.randint(0, 100), random.randint(0, 100), random.randint(0, 100), random.randint(0, 100) 

EVENT = 'detect'
COUNT_OBJ = 0
def Object_Detect():
    event_list = {
        "detect": read_classes,
        "luminous": ctl_luminous,
    }
    callback_list = {
        "detect": callback_detect,
        "luminous": callback_flash,
    }
    result = event_list[EVENT](callback_list[EVENT])
    if EVENT=='luminous':
        return result
    return 0

# event control
def ctl_luminous(callback):
    global EVENT, COUNT_OBJ
    lum = callback()
    print("E:{}, lum:{}, count:{}".format(EVENT, lum, COUNT_OBJ))
    if COUNT_OBJ==-1:
        # change the event
        EVENT = 'detect'
        return 0
    return lum
# https://medium.com/axinc-ai/detic-object-detection-and-segmentation-of-21k-classes-with-high-accuracy-49cba412b7d4
def read_classes(callback):
    global EVENT, COUNT_OBJ
    image_path = input('Enter Image Path:')
    image_path = os.path.realpath(image_path)
    print('Image path: {}'.format(image_path))
    if os.path.isfile(image_path)==False:
        print('Image file not found.')
        return 0
        
    result = input("Class name:(可以輸入多個要用空格隔開) \n(ex. webcam headphone)> ")
    classes = result.split(' ') if ' ' in result else [result]
    print('Classes: ', classes)
    if callback==None:
        print("Didn't have callback.")
        return len(classes)
    else:
        num = callback(image_path, classes)
        print('Instance numbers: ', num)
        COUNT_OBJ = num
        # change the event
        EVENT = 'luminous'
        return num

# call back function
def callback_detect(image_path, classes=['webcam', 'headphone']):
    _, num = detect_object(
                PREDICTOR,
                image_path,
                classes=classes,
                view=True)
    return num

_FLASH = False
def callback_flash():
    global _FLASH, COUNT_OBJ
    luminous =  100 if _FLASH==False else 0
    _FLASH = True if luminous==100 else False
    if _FLASH==True:
        COUNT_OBJ -= 1
    return luminous

def Dummy_Control(data:list):
    print(data[0])
