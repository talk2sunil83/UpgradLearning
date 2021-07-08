#  Another help : https://github.com/kitohe/android-cam-to-pc
import ssl
import cv2
import urllib.request
import numpy as np
import pyvirtualcam
# from ppadb.client import Client as AdbClient
# import time
# from icecream import ic

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# %%
url = 'http://192.168.0.201:8080/shot.jpg'
# while True:
#     imgResp = urllib.request.urlopen(url)
#     imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
#     img = cv2.imdecode(imgNp, -1)
#     cv2.imshow('test', img)
#     if ord('q') == cv2.waitKey(10):
#         exit(0)
# cv2.destroyAllWindows()
# %%
with pyvirtualcam.Camera(width=1080, height=1920, fps=30) as cam:

    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        # img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height = img.shape[0]
        width = img.shape[1]
        # with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
        cam.send(img)
        cam.sleep_until_next_frame()
        q = cv2.waitKey(1)
        if q == ord("q"):
            break

    cv2.destroyAllWindows()


# %%


# def connect():
#     client = AdbClient(host="127.0.0.1", port=5037)  # Default is "127.0.0.1" and 5037

#     devices = client.devices()

#     if len(devices) == 0:
#         print('No devices')
#         quit()

#     device = devices[0]

#     print(f'Connected to {device}')

#     return device, client


# # %%
# device, client = connect()
# # open up camera app
# device.shell('input keyevent 27')

# # wait 5 seconds
# time.sleep(2)

# # take a photo with volume up
# photo = device.shell('input keyevent 24')
# print('Taken a photo!')

# %%
