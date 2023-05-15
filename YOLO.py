# This code infers for WEB requests
import gevent
from gevent import monkey
monkey.patch_all(thread=False)
from gevent.pywsgi import WSGIServer
from flask import Flask, request

import urllib.request
import base64

import cv2
import torch

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel
import numpy as np
import sys
import os

import threading
lock = threading.Lock()

import mmap
import ctypes
import ctypes.wintypes
import win32api
import win32con
import win32gui
import win32ui

jcqbh = '1'
Q_device = ''
Q_weights = 'best.pt'
Q_imgsz = 640
dk = 7700


geshu = len(sys.argv)
if geshu < 6:
    print('参数不足，启动失败，2秒后自动退出')
    time.sleep(2)
    sys.exit()
else:
    jcqbh = sys.argv[1]
    Q_imgsz = int(sys.argv[2])
    dk = int(sys.argv[3])
    temp_device = sys.argv[4]
    if temp_device == 'aidmlm':
        Q_device = ''
    else:
        Q_device = temp_device
    Q_weights = sys.argv[5]

pid = os.getpid()

print('jcqbh', jcqbh)
print('weights', Q_weights)
print('imgsz', Q_imgsz)


t1 = time_synchronized()
# Initialize
set_logging()
device = select_device(Q_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(Q_weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(Q_imgsz, s=stride)  # check img_size
model = TracedModel(model, device, Q_imgsz)
if half:
    model.half()  # to FP16

# Get names
names = model.module.names if hasattr(model, 'module') else model.names
# print('所有的分类名：', names)

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

t2 = time_synchronized()
print(f'加载模型耗时{t2 - t1:.2f}s\n')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def bytesToMat(img):
    np_arr = np.frombuffer(bytearray(img), dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)



@torch.no_grad()
def yuce(img0):
    if hasattr(img0, 'shape') == False:
        return ''
    lock.acquire()

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.1, 0.45, classes=None, agnostic=True)

    # Process detections
    det = pred[0]
    xywh_list = ''
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()  #按比例修正了实际坐标

        for *xyxy, conf, cls in reversed(det):
            xywh = ""
            c = int(cls)
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            zxd = float(conf)
            xywh = str(c) + "," + str(x1) + "," + str(y1) + "," + str(w) + "," + str(h) + "," + str(zxd) + ',' + str(names[c]) + '|'
            xywh_list = xywh_list + xywh
    xywh_list = xywh_list.encode('utf-8')
    lock.release()
    return xywh_list


def Window_Shot(hwnd):
    if win32gui.IsWindow(hwnd) == False:
        hwnd = win32gui.GetDesktopWindow()
        MoniterDev = win32api.EnumDisplayMonitors(None, None)
        w = MoniterDev[0][2][2]
        h = MoniterDev[0][2][3]
    else:
        ret = win32gui.GetClientRect(hwnd)
        w = ret[2]
        h = ret[3]

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    signedIntsArray = saveBitMap.GetBitmapBits(True)
    im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
    im_opencv.shape = (h, w, 4)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return im_opencv


app = Flask(__name__)

@app.route('/pid', methods=['GET', 'POST'])
def YOLOv7_WEB_pid():
    return str(pid)

@app.route('/pic', methods=['GET', 'POST'])
def YOLOv7_WEB_pic():
    if request.data == b'':
        return ''
    try:
        val = request.data[0:int(request.content_length)]
        img = bytesToMat(val)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''

@app.route('/hwnd',methods=['GET', 'POST'])
def YOLOv7_WEB_hwnd():
    try:
        tp = request.get_data()
        sj = tp.decode('utf-8', "ignore")
        jb = int(float(sj)) # 是目标窗口句柄
        img = Window_Shot(jb)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''


@app.route('/base64', methods=['GET', 'POST'])
def YOLOv7_WEB_base64():
    try:
        tp = request.get_data()
        tp_wb = tp.decode('utf-8', "ignore")
        b_tp = base64.b64decode(tp_wb)
        img = bytesToMat(b_tp)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''


@app.route('/file',methods=['GET', 'POST'])
def YOLOv7_WEB_file():
    try:
        tp = request.get_data()
        tp_file = tp.decode('utf-8', "ignore")
        img = cv2.imdecode(np.fromfile(file=tp_file, dtype=np.uint8), -1)
        jieguo = yuce(img)
        return jieguo
    except:
        return ''




def qidong():
    print('使用端口号' + str(dk) + '创建WEB服务器')
    print('')

    WSGIServer(('0.0.0.0', dk), app, log=None).serve_forever() #B  , log=None   127.0.0.1  , log=None





SendMessage = ctypes.windll.user32.SendMessageA

class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_void_p)
    ]

class COPYDATASTRUCT2(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_char_p)
    ]

PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)



class Listener:
    def __init__(self):
        WindowName = "aidmlm.com" + jcqbh
        message_map = {
            win32con.WM_COPYDATA: self.OnCopyData
        }
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = message_map
        wc.lpszClassName = WindowName
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        classAtom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindow(
            classAtom,
            "aidmlm.com" + jcqbh,
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        print('pid', pid)
        print("hwnd", self.hwnd)
        threading.Thread(target=qidong).start()

    def OnCopyData(self, hwnd, msg, wparam, lparam):
        pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)
        s = ctypes.string_at(pCDS.contents.lpData, pCDS.contents.cbData).decode() # "utf-8", "ignore"
        cd = int(float(s))

        if wparam == 1:

            file_name = 'aidmlm.com' + jcqbh
            shmem = mmap.mmap(0, cd, file_name, mmap.ACCESS_WRITE)
            tp = shmem.read(cd)

            img = bytesToMat(tp)
            jieguo = yuce(img)


            jgcd = len(jieguo)
            shmem.seek(0)
            shmem.write(jieguo)

            shmem.close()

            return jgcd
        if wparam == 0:
            return pid
        if wparam == 2:
            jb = cd
            if win32gui.IsWindow(jb) == False:
                return 11

            img = Window_Shot(jb)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            jieguo = yuce(img)

            file_name = 'aidmlm.com' + jcqbh
            shmem = mmap.mmap(0, 51200, file_name, mmap.ACCESS_WRITE)

            jgcd = len(jieguo)
            shmem.seek(0)
            shmem.write(jieguo)
            shmem.close()

            return jgcd
        return 10


l = Listener()
win32gui.PumpMessages()
