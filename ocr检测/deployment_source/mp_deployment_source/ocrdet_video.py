import os
import ujson
import aicube
from media.sensor import *
from media.display import *
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import random
import gc

display_mode="lcd"
if display_mode=="lcd":
    DISPLAY_WIDTH = ALIGN_UP(800, 16)
    DISPLAY_HEIGHT = 480
else:
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080

OUT_RGB888P_WIDTH = ALIGN_UP(640, 16)
OUT_RGB888P_HEIGH = 640

root_path="/sdcard/kmodel/dl_ocr/"
config_path=root_path+"deploy_config.json"
deploy_conf={}
debug_mode=1

class ScopedTiming:
    def __init__(self, info="", enable_profile=True):
        self.info = info
        self.enable_profile = enable_profile

    def __enter__(self):
        if self.enable_profile:
            self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable_profile:
            elapsed_time = time.time_ns() - self.start_time
            print(f"{self.info} took {elapsed_time / 1000000:.2f} ms")

def read_deploy_config(config_path):
    # 打开JSON文件以进行读取deploy_config
    with open(config_path, 'r') as json_file:
        try:
            # 从文件中加载JSON数据
            config = ujson.load(json_file)

            # 打印数据（可根据需要执行其他操作）
            #print(config)
        except ValueError as e:
            print("JSON 解析错误:", e)
    return config

# 任务后处理

def get_pad_one_side_param(img_size):
    # 右padding或下padding
    dst_w = img_size[0]
    dst_h = img_size[1]

    ratio_w = dst_w / OUT_RGB888P_WIDTH
    ratio_h = dst_h / OUT_RGB888P_HEIGH
    if ratio_w < ratio_h:
        ratio = ratio_w
    else:
        ratio = ratio_h

    new_w = (int)(ratio * OUT_RGB888P_WIDTH)
    new_h = (int)(ratio * OUT_RGB888P_HEIGH)
    dw = (dst_w - new_w) / 2
    dh = (dst_h - new_h) / 2

    top = (int)(round(0))
    bottom = (int)(round(dh * 2 + 0.1))
    left = (int)(round(0))
    right = (int)(round(dw * 2 - 0.1))
    return [0, 0, 0, 0, top, bottom, left, right]

def ocr_detection():
    print("start")

    # 使用json读取内容初始化部署变量
    deploy_conf=read_deploy_config(config_path)
    kmodel_name=deploy_conf["kmodel_path"]
    mask_threshold=deploy_conf["mask_threshold"]
    box_threshold = deploy_conf["box_threshold"]
    img_size=deploy_conf["img_size"]
    # init kpu and load kmodel
    kpu = nn.kpu()
    ai2d = nn.ai2d()
    kpu.load_kmodel(root_path+kmodel_name)
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,
                   nn.ai2d_format.NCHW_FMT,
                   np.uint8, np.uint8)
    ai2d.set_pad_param(True, get_pad_one_side_param(img_size), 0, [0, 0, 0])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build([1,3,OUT_RGB888P_HEIGH,OUT_RGB888P_WIDTH], [1,3,img_size[0],img_size[1]])

    # 初始化并配置sensor
    sensor = Sensor()
    sensor.reset()
    # 设置镜像
    sensor.set_hmirror(False)
    # 设置翻转
    sensor.set_vflip(False)
    # 通道0直接给到显示VO，格式为YUV420
    sensor.set_framesize(width = DISPLAY_WIDTH, height = DISPLAY_HEIGHT)
    sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    # 通道2给到AI做算法处理，格式为RGB888
    sensor.set_framesize(width = OUT_RGB888P_WIDTH , height = OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)
    # 绑定通道0的输出到vo
    sensor_bind_info = sensor.bind_info(x = 0, y = 0, chn = CAM_CHN_ID_0)
    Display.bind_layer(**sensor_bind_info, layer = Display.LAYER_VIDEO1)
    if display_mode=="lcd":
        # 设置为ST7701显示，默认800x480
        Display.init(Display.ST7701, to_ide = True)
    else:
        # 设置为LT9611显示，默认1920x1080
        Display.init(Display.LT9611, to_ide = True)

    #创建OSD图像
    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)

    try:
        # media初始化
        MediaManager.init()
        # 启动sensor
        sensor.run()

        rgb888p_img = None
        ai2d_input_tensor = None
        data = np.ones((1,3,img_size[0],img_size[1]), dtype=np.uint8)
        ai2d_output_tensor = nn.from_numpy(data)
        while  True:
            with ScopedTiming("total",debug_mode > 0):
                rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)

                # for rgb888planar
                if rgb888p_img.format() == image.RGBP888:
                    ai2d_input = rgb888p_img.to_numpy_ref()
                    ai2d_input_tensor = nn.from_numpy(ai2d_input)
                    ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                    kpu.set_input_tensor(0, ai2d_output_tensor)
                    kpu.run()
                    results = []
                    for i in range(kpu.outputs_size()):
                        out_data = kpu.get_output_tensor(i)
                        result = out_data.to_numpy()
                        del out_data
                        results.append(result)
                    tmp = (ai2d_input.shape[0], ai2d_input.shape[1], ai2d_input.shape[2])
                    ai2d_input = ai2d_input.reshape((ai2d_input.shape[0], ai2d_input.shape[1] * ai2d_input.shape[2]))
                    ai2d_input = ai2d_input.transpose()
                    tmp2 = ai2d_input.copy()
                    tmp2 = tmp2.reshape((tmp[1], tmp[2], tmp[0]))
                    output_data = results[0][:, :, :, 0]
                    mp_list =  aicube.ocr_post_process(output_data.reshape(-1), tmp2.reshape(-1),
                                                        [img_size[0],img_size[1]],
                                                        [OUT_RGB888P_WIDTH, OUT_RGB888P_HEIGH], mask_threshold, box_threshold)
                    osd_img.clear()
                    if mp_list:
                        for j in mp_list:
                            for i in range(4):
                                x1 = j[1][(i * 2)]/OUT_RGB888P_WIDTH*DISPLAY_WIDTH
                                y1 = j[1][(i * 2 + 1)]/OUT_RGB888P_HEIGH*DISPLAY_HEIGHT
                                x2 = j[1][((i + 1) * 2) % 8]/OUT_RGB888P_WIDTH*DISPLAY_WIDTH
                                y2 = j[1][((i + 1) * 2 + 1) % 8]/OUT_RGB888P_HEIGH*DISPLAY_HEIGHT
                                osd_img.draw_line((int(x1), int(y1), int(x2), int(y2)), color=(255, 0, 0, 255),thickness=5)

                    Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
                gc.collect() #用于需要调用gc.mem_alloc()的内存
                rgb888p_img = None
    except Exception as e:
        print(f"An error occurred during buffer used: {e}")
    finally:
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        del ai2d_input_tensor
        del ai2d_output_tensor
        #停止摄像头输出
        sensor.stop()
        #去初始化显示设备
        Display.deinit()
        #释放媒体缓冲区
        MediaManager.deinit()
        gc.collect()
        time.sleep(1)
        nn.shrink_memory_pool()
    print("end")
    return 0


if __name__=="__main__":
    ocr_detection()
