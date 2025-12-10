# 部署包说明

部署包包含c++版本部署包(cpp_deployment_source.zip)和MicroPython版本部署包(mp_deployment_source.zip)。请分别解压查看对应的部署文件。

# C++版本部署流程

## 镜像说明

- 镜像版本：CanMV-K230_sdcard_v1.7_nncase_v2.9.0.img.gz
- 镜像下载链接：<https://kendryte-download.canaan-creative.com/k230/release/sdk_images/v1.7/k230_canmv_defconfig/CanMV-K230_sdcard_v1.7_nncase_v2.9.0.img.gz>
- 镜像烧录参考：<https://developer.canaan-creative.com/k230/zh/dev/01_software/board/K230_SDK_%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.html#id21> 5.1节'sd卡镜像烧写'

## 目录结构

部署包cpp_deployment_source.zip解压后得到如下目录及文件：

```
cpp_deployment_source
    |-example_code_k230 #示例代码，用于完成编译生成可执行文件main.elf
	    |-cmake
	    |-k230_deploy
	    |-build_app.sh
	    |-CMakeLists.txt
    |- *.kmodel #kmodel文件
    |- main.elf #支持canmv开发板的编译部署文件
    |- deploy_config.json #部署配置文件
    |- ... # 当前任务所需其他文件
```

## 编译示例代码

编译环境搭建请参考链接：https://github.com/kendryte/k230_sdk

如果您使用的是K230_CanMV开发板，在k230_sdk根目录下执行：

```
make CONF=k230_canmv_defconfig prepare_memory
make mpp
```

将example_code_k230文件拷贝到k230_sdk目录下的src/big/nncase下，执行build_app.sh

```
./build_app.sh
```

在example_code_k230下的k230_bin中得到编译的main.elf。默认编译出来的elf文件为支持canmv类型开发板的可执行文件。

如果您使用的是01Studio CanMV K230开发板，在k230_sdk根目录下执行：

```shell
make CONF=k230_canmv_01studio_defconfig prepare_memory
make mpp
```

将example_code_k230文件拷贝到k230_sdk目录下的src/big/nncase下，执行build_app.sh

```shell
#如果您想编译支持hdmi显示的可执行文件
./build_app.sh hdmi
#如果您想编译支持01studio MIPI LCD 800*480的屏幕的可执行文件
./build_app.sh lcd
```

在example_code_k230下的k230_bin中得到编译的main.elf。

**您在部署的过程中也可以直接使用已经帮您编译好的elf可执行文件main.elf/main_01_hdmi.elf/main_01_lcd.elf，注意：请选择对应的SDK版本编译SD卡镜像。**

## 开发板推理

在开发板sharefs目录下，新建一项目目录，如example。将得到的main.elf、*.kmodel、deploy_config.json、inference.jpg（待推理图片）拷贝到该新建目录下。在大核进入相同目录，执行推理命令：

```shell
# 静态图推理
./main.elf deploy_config.json inference.jpg 0
# deploy_config.json:部署配置文件
# inference.jpg:待推理图片，若使用摄像头则为None
# 0: 是否需要调试，0、1、2分别表示不调试、简单调试、详细调试
# 视频流推理
./main.elf deploy_config.json None 0
```

静态图推理在该目录下会生成result_*.jpg显示推理结果，摄像头推理会将结果实时显示在屏幕上。

**注意：不同任务需要拷贝不同文件，下面给出区别。**

| 任务类型   | 文件列表                                                     | 备注           |
| ---------- | ------------------------------------------------------------ | -------------- |
| 图像分类   | main.elf、*kmodel、deploy_config.json、test.jpg              |                |
| 目标检测   | main.elf、*kmodel、deploy_config.json、test.jpg              |                |
| 语义分割   | main.elf、*kmodel、deploy_config.json、test.jpg              |                |
| OCR检测    | main.elf、*kmodel、deploy_config.json、test.jpg              |                |
| OCR识别    | main.elf、*kmodel、deploy_config.json、test.jpg、dict.txt、dict_16.txt、Asci0816.zf、HZKf2424.hz | 仅支持单图推理 |
| OCR任务    | main.elf、*kmodel、deploy_config.json（两个，需改名，防止重名）、test.jpg、dict.txt、dict_16.txt、Asci0816.zf、HZKf2424.hz |                |
| 度量学习   | main.elf、*kmodel、deploy_config.json、test.jpg              |                |
| 多标签分类 | main.elf、*kmodel、deploy_config.json、test.jpg              |                |

**注意：不同任务的示例命令**

| 任务类型   | 示例命令                                                     | 备注           |
| ---------- | ------------------------------------------------------------ | -------------- |
| 图像分类   | ./main.elf deploy_config.json test.jpg 0<br />./main.elf deploy_config.json None 0 |                |
| 目标检测   | ./main.elf deploy_config.json test.jpg 0<br />./main.elf deploy_config.json None 0 |                |
| 语义分割   | ./main.elf deploy_config.json test.jpg 0<br />./main.elf deploy_config.json None 0 |                |
| OCR检测    | ./main.elf deploy_config.json test.jpg 0<br />./main.elf deploy_config.json None 0 |                |
| OCR识别    | ./main.elf deploy_config.json test.jpg 0                     | 仅支持单图推理 |
| OCR任务    | ./main.elf deploy_config_ocrdet.json deploy_config_ocrrec.json test.jpg 0<br />./main.elf deploy_config_ocrdet.json deploy_config_ocrrec.json None 0 |                |
| 度量学习   | ./main.elf deploy_config.json test.jpg 0<br />./main.elf deploy_config.json None 0 |                |
| 多标签分类 | ./main.elf deploy_config.json test.jpg 0<br />./main.elf deploy_config.json None 0 |                |

deploy_config.json中包含部署配置的参数，如果效果不好，请您重新训练模型或者修改阈值等参数。

# MicroPython 版本部署流程

## 镜像说明

- 镜像版本：CanMV-K230_micropython_v1.1-0-g5a6fc54_nncase_v2.9.0
- 镜像下载链接：<https://kendryte-download.canaan-creative.com/developer/k230/CanMV-K230_micropython_v1.1-0-g5a6fc54_nncase_v2.9.0.img.gz>
- 01studio镜像版本：CanMV-K230_01Studio_micropython_v1.1-0-g5a6fc54_nncase_v2.9.0
- 01studio镜像下载链接：<https://kendryte-download.canaan-creative.com/developer/k230/CanMV-K230_01Studio_micropython_v1.1-0-g5a6fc54_nncase_v2.9.0.img.gz>
- 镜像烧录参考：<https://developer.canaan-creative.com/k230_canmv/zh/main/zh/userguide/how_to_burn_firmware.html#id1>

## 目录结构

部署包mp_deployment_source.zip解压后得到如下目录及文件：

```
mp_deployment_source # micropython部署资源根目录
    |- *.kmodel # kmodel文件
    |- deploy_config.json # 部署配置文件
    |- *_image.py # 静态图推理micropython脚本
    |- *_video.py # 视频流推理micropython脚本
    |- *** # 任务相关其他文件
```

## 开发板推理流程

### 环境安装

下载CANMV-IDE并安装，下载地址 : <https://github.com/kendryte/canmv_ide/releases>

### 文件拷贝

您可以在IDE连接开发板的情况下，将mp_deployment_source文件夹拷贝到盘符`CanMV/sdcard/`目录下，静态图推理需要您自行拷贝一张测试图片，并命名为'test.jpg'；然后在IDE中选择`文件->打开文件->CanMV->sdcard->mp_deployment_source`目录，选择其中的推理脚本`**_image.py`或者`**_video.py`打开运行。

### 脚本运行

- 连接开发板，在CANMV-IDE中打开静态图推理脚本(##_image.py)或者视频流推理脚本(##_video.py)；
- 点击运行按钮；
- 部分任务的静态图推理结果会在IDE左上方小窗口显示，并保存为图片存储在当前目录下。
- 视频流推理结果会实时显示在屏幕上。