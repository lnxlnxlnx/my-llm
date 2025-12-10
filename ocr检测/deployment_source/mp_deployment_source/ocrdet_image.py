import aicube
import os
import ujson
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import gc
import utime

root_path="/sdcard/mp_deployment_source/"        # root_path要以/结尾
config_path=root_path+"deploy_config.json"
image_path=root_path+"test.jpg"
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

def read_img(img_path):
    img_data = image.Image(img_path)
    img_data_rgb888=img_data.to_rgb888()
    img_hwc=img_data_rgb888.to_numpy_ref()
    shape=img_hwc.shape
    img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
    img_tmp_trans = img_tmp.transpose()
    img_res=img_tmp_trans.copy()
    img_return=img_res.reshape((1,shape[2],shape[0],shape[1]))
    return img_return

def get_pad_one_side_param(img_size,src_size):
    # 右padding或下padding
    dst_w = img_size[0]
    dst_h = img_size[1]

    ratio_w = dst_w / src_size[3]
    ratio_h = dst_h / src_size[2]
    if ratio_w < ratio_h:
        ratio = ratio_w
    else:
        ratio = ratio_h

    new_w = (int)(ratio * src_size[3])
    new_h = (int)(ratio * src_size[2])
    dw = (dst_w - new_w) / 2
    dh = (dst_h - new_h) / 2

    top = (int)(round(0))
    bottom = (int)(round(dh * 2 + 0.1))
    left = (int)(round(0))
    right = (int)(round(dw * 2 - 0.1))
    return [0, 0, 0, 0, top, bottom, left, right]

# 读取deploy_config.json文件
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

def ocr_detection():
    print("--------------start-----------------")
    # 使用json读取内容初始化部署变量
    deploy_conf=read_deploy_config(config_path)
    kmodel_name=deploy_conf["kmodel_path"]
    mask_threshold=deploy_conf["mask_threshold"]
    box_threshold = deploy_conf["box_threshold"]
    img_size=deploy_conf["img_size"]

    # ai2d输入输出初始化
    ai2d_input = read_img(image_path)
    ai2d_input_tensor = nn.from_numpy(ai2d_input)
    ai2d_input_shape=ai2d_input.shape
    data = np.ones((1,3,img_size[0],img_size[1]),dtype=np.uint8)
    ai2d_out_tensor = nn.from_numpy(data)

    # init kpu and load kmodel
    kpu = nn.kpu()
    ai2d = nn.ai2d()
    kpu.load_kmodel(root_path + kmodel_name)
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,
                   nn.ai2d_format.NCHW_FMT,
                   np.uint8, np.uint8)
    ai2d.set_pad_param(True, get_pad_one_side_param(img_size,ai2d_input_shape), 0, [0, 0, 0])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build([1, 3, ai2d_input_shape[2], ai2d_input_shape[3]], [1, 3, img_size[0], img_size[1]])
    with ScopedTiming("total",debug_mode > 0):
        ai2d_builder.run(ai2d_input_tensor, ai2d_out_tensor)
        kpu.set_input_tensor(0, ai2d_out_tensor)
        kpu.run()
        del ai2d_input_tensor
        del ai2d_out_tensor
        # 获取分类结果
        results = []
        for i in range(kpu.outputs_size()):
            data = kpu.get_output_tensor(i)
            result = data.to_numpy()
            del data
            results.append(result)
        tmp = (ai2d_input.shape[1], ai2d_input.shape[2], ai2d_input.shape[3])
        ai2d_input = ai2d_input.reshape((ai2d_input.shape[1], ai2d_input.shape[2] * ai2d_input.shape[3]))
        ai2d_input = ai2d_input.transpose()
        tmp2 = ai2d_input.copy()
        tmp2 = tmp2.reshape((tmp[1], tmp[2], tmp[0]))
        output_data = results[0][:, :, :, 0]
        del ai2d_input
        mp_list = aicube.ocr_post_process(output_data.reshape(-1), tmp2.reshape(-1),
                                          [img_size[0], img_size[1]],
                                          [ai2d_input_shape[3], ai2d_input_shape[2]], mask_threshold, box_threshold)

        image_draw = image.Image(image_path).to_rgb565()
        if mp_list:
            for j in mp_list:
                for i in range(4):
                    x1 = j[1][(i * 2)]
                    y1 = j[1][(i * 2 + 1)]
                    x2 = j[1][((i + 1) * 2) % 8]
                    y2 = j[1][((i + 1) * 2 + 1) % 8]
                    image_draw.draw_line((int(x1), int(y1), int(x2), int(y2)), color=(255, 0, 0, 255), thickness=5)
        image_draw.compress_for_ide()
        image_draw.save(root_path + "ocrdet_result.jpg")
        del ai2d
        del ai2d_builder
        del kpu
        gc.collect()
    print("---------------end------------------")
    nn.shrink_memory_pool()


if __name__=="__main__":
    nn.shrink_memory_pool()
    ocr_detection()
