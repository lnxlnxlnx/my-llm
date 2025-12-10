# 整合OCR识别与TCP通信
from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from media.sensor import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import aicube
import random
import gc
import sys
import socket
import network

# ------------------- OCR相关类定义 -------------------
class OCRDetectionApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,mask_threshold=0.5,box_threshold=0.2,rgb888p_size=[224,224],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.model_input_size=model_input_size
        self.mask_threshold=mask_threshold
        self.box_threshold=box_threshold
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    def config_preprocess(self,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            top,bottom,left,right=self.get_padding_param()
            self.ai2d.pad([0,0,0,0,top,bottom,left,right], 0, [0,0,0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            hwc_array=self.chw2hwc(self.cur_img)
            det_boxes = aicube.ocr_post_process(results[0][:,:,:,0].reshape(-1), hwc_array.reshape(-1),self.model_input_size,self.rgb888p_size, self.mask_threshold, self.box_threshold)
            return det_boxes

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        input_width = self.rgb888p_size[0]
        input_high = self.rgb888p_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * input_width)
        new_h = (int)(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return  top, bottom, left, right

    def chw2hwc(self,features):
        ori_shape = (features.shape[0], features.shape[1], features.shape[2])
        c_hw_ = features.reshape((ori_shape[0], ori_shape[1] * ori_shape[2]))
        hw_c_ = c_hw_.transpose()
        new_array = hw_c_.copy()
        hwc_array = new_array.reshape((ori_shape[1], ori_shape[2], ori_shape[0]))
        del c_hw_
        del hw_c_
        del new_array
        return hwc_array

class OCRRecognitionApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,dict_path,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.model_input_size=model_input_size
        self.dict_path=dict_path
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.dict_word=None
        self.read_dict()
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.RGB_packed,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    def config_preprocess(self,input_image_size=None,input_np=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            top,bottom,left,right=self.get_padding_param(ai2d_input_size,self.model_input_size)
            self.ai2d.pad([0,0,0,0,top,bottom,left,right], 0, [0,0,0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([input_np.shape[0],input_np.shape[1],input_np.shape[2],input_np.shape[3]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            preds = np.argmax(results[0], axis=2).reshape((-1))
            output_txt = ""
            for i in range(len(preds)):
                if preds[i] != (len(self.dict_word) - 1) and (not (i > 0 and preds[i - 1] == preds[i])):
                    output_txt = output_txt + self.dict_word[preds[i]]
            return output_txt

    def get_padding_param(self,src_size,dst_size):
        dst_w = dst_size[0]
        dst_h = dst_size[1]
        input_width = src_size[0]
        input_high = src_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * input_width)
        new_h = (int)(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return  top, bottom, left, right

    def read_dict(self):
        if self.dict_path!="":
            with open(self.dict_path, 'r') as file:
                line_one = file.read(100000)
                line_list = line_one.split("\r\n")
            self.dict_word = {num: char.replace("\r", "").replace("\n", "") for num, char in enumerate(line_list)}

class OCRDetRec:
    def __init__(self,ocr_det_kmodel,ocr_rec_kmodel,det_input_size,rec_input_size,dict_path,mask_threshold=0.25,box_threshold=0.3,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        self.ocr_det_kmodel=ocr_det_kmodel
        self.ocr_rec_kmodel=ocr_rec_kmodel
        self.det_input_size=det_input_size
        self.rec_input_size=rec_input_size
        self.dict_path=dict_path
        self.mask_threshold=mask_threshold
        self.box_threshold=box_threshold
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.ocr_det=OCRDetectionApp(self.ocr_det_kmodel,model_input_size=self.det_input_size,mask_threshold=self.mask_threshold,box_threshold=self.box_threshold,rgb888p_size=self.rgb888p_size,display_size=self.display_size,debug_mode=0)
        self.ocr_rec=OCRRecognitionApp(self.ocr_rec_kmodel,model_input_size=self.rec_input_size,dict_path=self.dict_path,rgb888p_size=self.rgb888p_size,display_size=self.display_size)
        self.ocr_det.config_preprocess()

    def run(self,input_np):
        det_res=self.ocr_det.run(input_np)
        boxes=[]
        ocr_res=[]
        for det in det_res:
            self.ocr_rec.config_preprocess(input_image_size=[det[0].shape[2],det[0].shape[1]],input_np=det[0])
            ocr_str=self.ocr_rec.run(det[0])
            ocr_res.append(ocr_str)
            boxes.append(det[1])
            gc.collect()
        return boxes,ocr_res

    def draw_result(self,pl,det_res,rec_res):
        pl.osd_img.clear()
        if det_res:
            for j in range(len(det_res)):
                for i in range(4):
                    x1 = det_res[j][(i * 2)] / self.rgb888p_size[0] * self.display_size[0]
                    y1 = det_res[j][(i * 2 + 1)] / self.rgb888p_size[1] * self.display_size[1]
                    x2 = det_res[j][((i + 1) * 2) % 8] / self.rgb888p_size[0] * self.display_size[0]
                    y2 = det_res[j][((i + 1) * 2 + 1) % 8] / self.rgb888p_size[1] * self.display_size[1]
                    pl.osd_img.draw_line((int(x1), int(y1), int(x2), int(y2)), color=(255, 0, 0, 255),thickness=5)
                pl.osd_img.draw_string_advanced(int(x1),int(y1),32,rec_res[j],color=(0,0,255))

# ------------------- 网络相关函数 -------------------
def network_use_wlan(is_wlan=True):
    if is_wlan:
        sta = network.WLAN(0)
        sta.connect("lnx", "888888880")
        print("WLAN status:", sta.status())
        while sta.ifconfig()[0] == '0.0.0.0':
            os.exitpoint()
        print("WLAN config:", sta.ifconfig())
        ip = sta.ifconfig()[0]
        return ip, sta
    else:
        a = network.LAN()
        if not a.active():
            raise RuntimeError("LAN interface is not active.")
        a.ifconfig("dhcp")
        print("LAN config:", a.ifconfig())
        ip = a.ifconfig()[0]
        return ip, a

# ------------------- TCP服务器类 -------------------
class TCPServer:
    def __init__(self, ip, port=8888):
        self.ip = ip
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_stream = None
        self.connected = False

    def start(self):
        # 创建socket
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 绑定地址
        ai = socket.getaddrinfo("0.0.0.0", self.port)
        addr = ai[0][-1]
        self.server_socket.bind(addr)

        # 开始监听
        self.server_socket.listen(5)
        print(f"TCP Server listening on {self.ip}:{self.port}")
        print("等待PC客户端连接...")

    def accept_client(self):
        # 等待客户端连接（处理EAGAIN错误）
        while True:
            try:
                res = self.server_socket.accept()
                self.client_socket = res[0]
                self.client_addr = res[1]
                # 使用阻塞模式
                self.client_socket.setblocking(True)
                self.client_stream = self.client_socket
                self.connected = True
                print(f"PC客户端已连接: {self.client_addr}")
                break
            except Exception as e:
                if e.errno == 11:  # EAGAIN错误，重试
                    os.exitpoint()
                    continue
                else:
                    raise e

    def send_data(self, data):
        if self.connected and self.client_stream:
            try:
                self.client_stream.write(data.encode() + b"\n")
                return True
            except Exception as e:
                print(f"发送数据失败: {e}")
                self.close_client()
                return False
        return False

    def close_client(self):
        if self.client_stream:
            self.client_stream.close()
        self.client_socket = None
        self.client_stream = None
        self.connected = False
        print("客户端连接已关闭")

    def close(self):
        self.close_client()
        if self.server_socket:
            self.server_socket.close()
        print("服务器已关闭")

# ------------------- 主程序 -------------------
if __name__=="__main__":
    # 显示配置
    display="lcd3_5"
    if display=="hdmi":
        display_mode='hdmi'
        display_size=[1920,1080]
    elif display=="lcd3_5":
        display_mode= 'st7701'
        display_size=[800,480]
    elif display=="lcd2_4":
        display_mode= 'st7701'
        display_size=[640,480]

    rgb888p_size=[640,360]

    # OCR模型配置
#    ocr_det_kmodel_path="/sdcard/kmodel/dl_ocr/ocr_det.kmodel"
#    ocr_rec_kmodel_path="/sdcard/kmodel/dl_ocr_rec/ocr_rec.kmodel"

    ocr_det_kmodel_path="/sdcard/examples/kmodel/ocr_det_int16.kmodel"
    ocr_rec_kmodel_path="/sdcard/examples/kmodel/ocr_rec_int16.kmodel"
    dict_path="/sdcard/examples/utils/dict.txt"
    ocr_det_input_size=[640,640]
    ocr_rec_input_size=[512,32]
    mask_threshold=0.25
    box_threshold=0.3

    # 初始化OCR
    pl=PipeLine(rgb888p_size=rgb888p_size,display_size=display_size,display_mode=display_mode)
    if display =="lcd2_4":
        pl.create(Sensor(width=1280, height=960))
    else:
        pl.create(Sensor(width=1920, height=1080))
    ocr=OCRDetRec(ocr_det_kmodel_path,ocr_rec_kmodel_path,det_input_size=ocr_det_input_size,rec_input_size=ocr_rec_input_size,dict_path=dict_path,mask_threshold=mask_threshold,box_threshold=box_threshold,rgb888p_size=rgb888p_size,display_size=display_size)

    # 初始化网络和TCP服务器
    ip, wlan = network_use_wlan(True)
    tcp_server = TCPServer(ip)
    tcp_server.start()

    # 主循环
    clock = time.clock()
    counter = 0

    try:
        while True:
            # 如果没有客户端连接，等待连接
            if not tcp_server.connected:
                tcp_server.accept_client()

            # 获取图像并进行OCR识别
            img = pl.get_frame()
            det_res, rec_res = ocr.run(img)

            # 绘制识别结果
            ocr.draw_result(pl, det_res, rec_res)
            pl.show_image()

            # 打印并发送识别结果
            if rec_res:
                ocr_text = " | ".join(rec_res)
#                print(f"\n识别结果 #{counter}: {ocr_text}")
#                print(f"检测框: {det_res}")
#                print(f"帧率: {clock.fps():.2f}")
                print(f"{ocr_text}")

                # 发送数据到PC
                #send_data = f"OCR识别结果 #{counter}: {ocr_text}\n检测框坐标: {det_res}"
                send_data = f"{ocr_text}\n"
                tcp_server.send_data(send_data)

            counter += 1
            gc.collect()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
    finally:
        tcp_server.close()
        print("资源已释放，程序退出")
