/* Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <thread>
#include <map>
#include "utils.h"
#include "vi_vo.h"
#include "classification.h"
#include "anchorbase_det.h"
#include "anchorfree_det.h"
#include "gfl_det.h"
#include "multilabel_cls.h"
#include "ocr_box.h"
#include "ocr_reco.h"
#include "metriclearning.h"
#include "segmentation.h"
#include "anomaly_det.h"

using std::cerr;
using std::cout;
using std::endl;
using namespace std;


std::atomic<bool> isp_stop(false);

std::map<string, int> modeltype;

void init()
{
    modeltype.insert(pair<string, int>("AnchorBaseDet", 0));        // 检测模型
    modeltype.insert(pair<string, int>("AnchorFreeDet", 1));
    modeltype.insert(pair<string, int>("FreeDet",       2));
    modeltype.insert(pair<string, int>("GFLDet",        3));        // 检测模型

    modeltype.insert(pair<string, int>("can1",          10));        // 分类模型
    modeltype.insert(pair<string, int>("can2",          11));
    modeltype.insert(pair<string, int>("can5",          12));
    modeltype.insert(pair<string, int>("can6",          13));
    modeltype.insert(pair<string, int>("can7",          14));
    modeltype.insert(pair<string, int>("can8",          15));
    modeltype.insert(pair<string, int>("can9",          16));       // 分类模型

    modeltype.insert(pair<string, int>("DeepNet",       20));       // 分割模型
    modeltype.insert(pair<string, int>("EDNet",         21));

    modeltype.insert(pair<string, int>("OCR_DNet",      30));       // ocr-box模型

    modeltype.insert(pair<string, int>("OCR_RLNet",      40));       // ocr-reco模型
    modeltype.insert(pair<string, int>("OCR_RCNet",       41));

    modeltype.insert(pair<string, int>("PatchDet",      50));       // 异常检测模型

    modeltype.insert(pair<string, int>("multilabel",      60));       // 多标签分类

    modeltype.insert(pair<string, int>("metric_learning",      70));       // 多标签分类

}


void print_usage()
{
    cout << "单模型推理时传参说明：" << "<config_file> <input_mode> <debug_mode>" << endl
         << "Options:" << endl
         << "  config_file     部署所用json配置文件,默认为 deploy_config.json\n"
         << "  input_mode      本地图片(图片路径)/ 摄像头(None) \n"
         << "  debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;

    cout << "双模型推理时传参说明：" << "<config_file_1> <config_file_2> <input_mode> <debug_mode>" << endl
         << "Options:" << endl
         << "  config_file_1   部署所用json配置文件,为检测模型的 deploy_config1.json\n"
         << "  config_file_2   部署所用json配置文件,为识别模型的 deploy_config2.json\n"
         << "  input_mode      本地图片(图片路径)/ 摄像头(None) \n"
         << "  debug_mode      是否需要调试，0、1、2分别表示不调试、简单调试、详细调试\n"
         << "\n"
         << endl;
}

void image_proc_cls(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    Classification cls(args, atoi(argv[3]));

    cls.pre_process(ori_img);

    cls.inference();

    vector<cls_res> results;
    cls.post_process(results);

    Utils::draw_cls_res(ori_img,results);
    cv::imwrite("result_cls.jpg", ori_img);
    
}

void image_proc_ob_det(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;
    vector<ob_det_res> results;

    if (args.model_type == "AnchorBaseDet")
    {
        AnchorBaseDet ob_det(args, atoi(argv[3]));
        ob_det.pre_process(ori_img);
        ob_det.inference();
        ob_det.post_process({ori_w, ori_h}, results);
    }
    else if (args.model_type == "GFLDet")
    {
        GFLDet ob_det(args, atoi(argv[3]));
        ob_det.pre_process(ori_img);
        ob_det.inference();
        ob_det.post_process({ori_w, ori_h}, results);
    }
    else if (args.model_type == "FreeDet" || args.model_type == "AnchorFreeDet")
    {
        AnchorFreeDet ob_det(args, atoi(argv[3]));
        ob_det.pre_process(ori_img);
        ob_det.inference();
        ob_det.post_process({ori_w, ori_h}, results);
    }
    else
    {
        std::cerr << "不支持此类的检测模型";
        return ;
    }
    
    Utils::draw_ob_det_res(ori_img,results);
    cv::imwrite("result_ob_det.jpg", ori_img);
}

void image_proc_mlcls(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    MultilabelCls mlcls(args, atoi(argv[3]));

    mlcls.pre_process(ori_img);

    mlcls.inference();

    vector<multi_lable_res> results;
    mlcls.post_process(results);

    Utils::draw_mlcls_res(ori_img,results);
    cv::imwrite("result_mlcls.jpg", ori_img);  
}

void image_proc_ocr_det(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    OCRBox ocrbox(args, 0, atoi(argv[3]));

    ocrbox.pre_process(ori_img);

    ocrbox.inference();

    vector<ocr_det_res> results;
    ocrbox.post_process({ori_w, ori_h},results);

    Utils::draw_ocr_det_res(ori_img,results);
    cv::imwrite("result_ocr_det.jpg", ori_img);  
}

void image_proc_ocr_rec(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    OCRReco ocrrec(args, atoi(argv[3]));

    ocrrec.pre_process(ori_img);

    ocrrec.inference();

    vector<unsigned char> results;
    ocrrec.post_process(results);

    Utils::draw_ocr_rec_res(ori_img,results);
    cv::imwrite("result_ocr_rec.jpg", ori_img);  
}

void image_proc_ocr_det_rec(config_args args1, config_args args2, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[3]);
    cv::Mat draw_img = ori_img.clone();
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    OCRBox ocrbox(args1, 0, atoi(argv[4]));
    OCRReco ocrrec(args2, atoi(argv[4]));

    ocrbox.pre_process(ori_img);

    ocrbox.inference();

    vector<ocr_det_res> results_box;
    ocrbox.post_process({ori_w, ori_h}, results_box);

    for(int i = 0; i < results_box.size(); i++)
    {
        vector<Point> vec;
        vector<Point2f> sort_vtd(4);
        vec.clear();
        for(int j = 0; j < 4; j++)
        {
            vec.push_back(results_box[i].vertices[j]);
        }
        cv::RotatedRect rect = cv::minAreaRect(vec);
        cv::Point2f ver[4];
        rect.points(ver);
        cv::Mat crop;
        Utils::warppersp(ori_img, crop, results_box[i], sort_vtd);

        ocrrec.pre_process(crop);
        ocrrec.inference();

        vector<unsigned char> results_rec;
        ocrrec.post_process(results_rec);
        Utils::draw_ocr_text(int(sort_vtd[3].x), int(sort_vtd[3].y),draw_img,results_rec);
    }

    Utils::draw_ocr_det_res(draw_img, results_box);
    cv::imwrite("result_ocr_box_rec.jpg", draw_img);
}

void image_proc_ml(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    Metriclearning ml(args, atoi(argv[3]));

    ml.pre_process(ori_img);

    ml.inference();

    std::string name = "result_ml.bin";
    ml.post_process(name.c_str());
}

void image_proc_seg(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;

    Segmentation seg(args, atoi(argv[3]));

    seg.pre_process(ori_img);

    seg.inference();

    cv::Mat pred_color;
    seg.post_process(pred_color);
    cv::imwrite("result_seg.jpg", pred_color);
}


void image_proc_anomaly(config_args args, char *argv[])
{
    cv::Mat ori_img = cv::imread(argv[2]);
    int ori_w = ori_img.cols;
    int ori_h = ori_img.rows;
    AnomalyDet ad(args, atoi(argv[3]));
    ad.pre_process(ori_img);
    ad.inference();
    vector<anomaly_res> results;
    ad.post_process(results);
    Utils::draw_anomaly_res(ori_img, results);
    cv::imwrite("anomaly_detection_result.jpg", ori_img);
    
}

void video_proc_cls(config_args args, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    Classification cls(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));

    vector<cls_res> results;

    while (!isp_stop)
    {
        ScopedTiming st("total time", atoi(argv[3]));

        {
            ScopedTiming st("read capture", atoi(argv[3]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0 , sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }
            

        {
            ScopedTiming st("isp copy", atoi(argv[3]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        results.clear();

        cls.pre_process();
        cls.inference();

        cls.post_process(results);

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        {   
            #if defined(CONFIG_BOARD_K230D_CANMV)
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                Utils::draw_cls_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }
            #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
            {
                #if defined(STUDIO_HDMI)
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    Utils::draw_cls_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    Utils::draw_cls_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #endif
            }
            #else
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                Utils::draw_cls_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }
            #endif
        }


        {
            ScopedTiming st("osd copy", atoi(argv[3]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            //显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
            // printf("kd_mpi_vo_chn_insert_frame success \n");

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();


    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
}

void video_proc_ob_det(config_args args, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    vector<ob_det_res> results;

    if (args.model_type == "AnchorBaseDet")
    {
        AnchorBaseDet ob_det(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));
        while (!isp_stop)
        {
            ScopedTiming st("total time", atoi(argv[3]));

            {
                ScopedTiming st("read capture", atoi(argv[3]));
                // VICAP_CHN_ID_1 out rgb888p
                memset(&dump_info, 0 , sizeof(k_video_frame_info));
                ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
                if (ret) {
                    printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                    continue;
                }
            }
                

            {
                ScopedTiming st("isp copy", atoi(argv[3]));
                // 从vivcap中读取一帧图像到dump_info
                auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
                memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
                kd_mpi_sys_munmap(vbvaddr, size);
            }

            results.clear();

            ob_det.pre_process();
            ob_det.inference();

            ob_det.post_process({SENSOR_WIDTH, SENSOR_HEIGHT},results);

            cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

            {
                #if defined(CONFIG_BOARD_K230D_CANMV)
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    Utils::draw_ob_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
                {

                    #if defined(STUDIO_HDMI)
                    {
                        ScopedTiming st("osd draw", atoi(argv[3]));
                        Utils::draw_ob_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    }
                    #else
                    {
                        ScopedTiming st("osd draw", atoi(argv[3]));
                        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                        Utils::draw_ob_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                    }
                    #endif
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    Utils::draw_ob_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
                #endif
            }


            {
                ScopedTiming st("osd copy", atoi(argv[3]));
                memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
                //显示通道插入帧
                kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
                // printf("kd_mpi_vo_chn_insert_frame success \n");

                ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
                if (ret) {
                    printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
                }
            }
        }
    }
    else if (args.model_type == "GFLDet")
    {
        GFLDet ob_det(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));
        while (!isp_stop)
        {
            ScopedTiming st("total time", atoi(argv[3]));

            {
                ScopedTiming st("read capture", atoi(argv[3]));
                // VICAP_CHN_ID_1 out rgb888p
                memset(&dump_info, 0 , sizeof(k_video_frame_info));
                ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
                if (ret) {
                    printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                    continue;
                }
            }
                

            {
                ScopedTiming st("isp copy", atoi(argv[3]));
                // 从vivcap中读取一帧图像到dump_info
                auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
                memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
                kd_mpi_sys_munmap(vbvaddr, size);
            }

            results.clear();

            ob_det.pre_process();
            ob_det.inference();

            ob_det.post_process({SENSOR_WIDTH, SENSOR_HEIGHT},results);

            cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

            {
                #if defined(CONFIG_BOARD_K230D_CANMV)
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    Utils::draw_ob_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
                {

                    #if defined(STUDIO_HDMI)
                    {
                        ScopedTiming st("osd draw", atoi(argv[3]));
                        Utils::draw_ob_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    }
                    #else
                    {
                        ScopedTiming st("osd draw", atoi(argv[3]));
                        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                        Utils::draw_ob_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                    }
                    #endif
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    Utils::draw_ob_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
                #endif
            }


            {
                ScopedTiming st("osd copy", atoi(argv[3]));
                memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
                //显示通道插入帧
                kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
                // printf("kd_mpi_vo_chn_insert_frame success \n");

                ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
                if (ret) {
                    printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
                }
            }
        }
    }
    else if (args.model_type == "FreeDet" || args.model_type == "AnchorFreeDet")
    {
        AnchorFreeDet ob_det(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));
        while (!isp_stop)
        {
            ScopedTiming st("total time", atoi(argv[3]));

            {
                ScopedTiming st("read capture", atoi(argv[3]));
                // VICAP_CHN_ID_1 out rgb888p
                memset(&dump_info, 0 , sizeof(k_video_frame_info));
                ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
                if (ret) {
                    printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                    continue;
                }
            }
                

            {
                ScopedTiming st("isp copy", atoi(argv[3]));
                // 从vivcap中读取一帧图像到dump_info
                auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
                memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
                kd_mpi_sys_munmap(vbvaddr, size);
            }

            results.clear();

            ob_det.pre_process();
            ob_det.inference();

            ob_det.post_process({SENSOR_WIDTH, SENSOR_HEIGHT},results);

            cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

            {
                #if defined(CONFIG_BOARD_K230D_CANMV)
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    Utils::draw_ob_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
                {

                    #if defined(STUDIO_HDMI)
                    {
                        ScopedTiming st("osd draw", atoi(argv[3]));
                        Utils::draw_ob_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    }
                    #else
                    {
                        ScopedTiming st("osd draw", atoi(argv[3]));
                        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                        Utils::draw_ob_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                        cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                    }
                    #endif
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    Utils::draw_ob_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
                #endif
            }


            {
                ScopedTiming st("osd copy", atoi(argv[3]));
                memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
                //显示通道插入帧
                kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
                // printf("kd_mpi_vo_chn_insert_frame success \n");

                ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
                if (ret) {
                    printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
                }
            }
        }
    }
    else
    {
        std::cerr << "不支持此类的检测模型";
        return ;
    }
    vo_osd_release_block();
    vivcap_stop();


    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
}

void video_proc_mlcls(config_args args, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    MultilabelCls mlcls(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));

    vector<multi_lable_res> results;

    while (!isp_stop)
    {
        ScopedTiming st("total time", atoi(argv[3]));

        {
            ScopedTiming st("read capture", atoi(argv[3]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0 , sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }
            

        {
            ScopedTiming st("isp copy", atoi(argv[3]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        results.clear();

        mlcls.pre_process();
        mlcls.inference();

        mlcls.post_process(results);

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        {

            #if defined(CONFIG_BOARD_K230D_CANMV)
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                Utils::draw_mlcls_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }
            #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
            {

                #if defined(STUDIO_HDMI)
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    Utils::draw_mlcls_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    Utils::draw_mlcls_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #endif
            }
            #else
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                Utils::draw_mlcls_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }
            #endif
        }


        {
            ScopedTiming st("osd copy", atoi(argv[3]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            //显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
            // printf("kd_mpi_vo_chn_insert_frame success \n");

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();


    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }   
}

void video_proc_ocr_det(config_args args, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    OCRBox ocrbox(args, 0, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));

    vector<ocr_det_res> results;

    while (!isp_stop)
    {
        ScopedTiming st("total time", atoi(argv[3]));

        {
            ScopedTiming st("read capture", atoi(argv[3]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0 , sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }
            

        {
            ScopedTiming st("isp copy", atoi(argv[3]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        results.clear();

        ocrbox.pre_process();
        ocrbox.inference();

        ocrbox.post_process({SENSOR_WIDTH, SENSOR_HEIGHT},results);

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        {   
            #if defined(CONFIG_BOARD_K230D_CANMV)
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                Utils::draw_ocr_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }
            #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
            {

                #if defined(STUDIO_HDMI)
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    Utils::draw_ocr_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    Utils::draw_ocr_det_res(osd_frame, results, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #endif
            }
            #else
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                Utils::draw_ocr_det_res(osd_frame, results, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }
            #endif
        }


        {
            ScopedTiming st("osd copy", atoi(argv[3]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            //显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
            // printf("kd_mpi_vo_chn_insert_frame success \n");

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();


    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }   
}

void video_proc_ocr_det_rec(config_args args1, config_args args2, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL;       //osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    OCRBox ocrbox(args1, 0, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[4]));
    OCRReco ocrrec(args2, atoi(argv[4]));

    vector<ocr_det_res> results_box;

    while (!isp_stop)
    {
        ScopedTiming st("total time", 1);

        {
            ScopedTiming st("read capture", atoi(argv[4]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0 , sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }
            

        {
            ScopedTiming st("isp copy", atoi(argv[4]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3);  // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        results_box.clear();

        ocrbox.pre_process();
        ocrbox.inference();

        ocrbox.post_process({SENSOR_WIDTH, SENSOR_HEIGHT}, results_box);

    
        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        #if defined(CONFIG_BOARD_K230D_CANMV)
        {
            ScopedTiming st("osd draw", atoi(argv[3]));
            cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);

            int matsize = SENSOR_WIDTH * SENSOR_HEIGHT;
            cv::Mat ori_img;
            cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
            cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
            cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
            std::vector<cv::Mat> sensor_rgb;
            sensor_rgb.push_back(ori_img_B);
            sensor_rgb.push_back(ori_img_G);
            sensor_rgb.push_back(ori_img_R);
            cv::merge(sensor_rgb, ori_img);

            for(int i = 0; i < results_box.size(); i++)
            {
                vector<Point> vec;
                vector<Point2f> sort_vtd(4);
                vec.clear();
                for(int j = 0; j < 4; j++)
                {
                    vec.push_back(results_box[i].vertices[j]);
                }
                cv::RotatedRect rect = cv::minAreaRect(vec);
                cv::Point2f ver[4];
                rect.points(ver);

                cv::Mat crop;
                Utils::warppersp(ori_img, crop, results_box[i], sort_vtd);

                ocrrec.pre_process(crop);
                ocrrec.inference();

                vector<unsigned char> results_rec;
                ocrrec.post_process(results_rec);
                Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results_rec,{osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }

            {
                ScopedTiming st("osd draw", atoi(argv[4]));
                // cv::putText(osd_frame, "This !!!", {50, 600}, 4, 4, cv::Scalar(255, 255, 255, 255), 4, 8, 0);
                // cv::rectangle(osd_frame, cv::Rect(osd_width-(200 + 200), osd_height-(700 + 700) , 200, 700), cv::Scalar(255,255, 255, 255), 2, 2, 0);
                Utils::draw_ocr_det_res(osd_frame, results_box, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }
            cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
        }
        #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
        {

            #if defined(STUDIO_HDMI)
            {
                int matsize = SENSOR_WIDTH * SENSOR_HEIGHT;
                cv::Mat ori_img;
                cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
                cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
                cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
                std::vector<cv::Mat> sensor_rgb;
                sensor_rgb.push_back(ori_img_B);
                sensor_rgb.push_back(ori_img_G);
                sensor_rgb.push_back(ori_img_R);
                cv::merge(sensor_rgb, ori_img);

                for(int i = 0; i < results_box.size(); i++)
                {
                    vector<Point> vec;
                    vector<Point2f> sort_vtd(4);
                    vec.clear();
                    for(int j = 0; j < 4; j++)
                    {
                        vec.push_back(results_box[i].vertices[j]);
                    }
                    cv::RotatedRect rect = cv::minAreaRect(vec);
                    cv::Point2f ver[4];
                    rect.points(ver);

                    cv::Mat crop;
                    Utils::warppersp(ori_img, crop, results_box[i], sort_vtd);

                    ocrrec.pre_process(crop);
                    ocrrec.inference();

                    vector<unsigned char> results_rec;
                    ocrrec.post_process(results_rec);
                    Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results_rec,{osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }

                {
                    ScopedTiming st("osd draw", atoi(argv[4]));
                    // cv::putText(osd_frame, "This !!!", {50, 600}, 4, 4, cv::Scalar(255, 255, 255, 255), 4, 8, 0);
                    // cv::rectangle(osd_frame, cv::Rect(osd_width-(200 + 200), osd_height-(700 + 700) , 200, 700), cv::Scalar(255,255, 255, 255), 2, 2, 0);
                    Utils::draw_ocr_det_res(osd_frame, results_box, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }
            }
            #else
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                
                int matsize = SENSOR_WIDTH * SENSOR_HEIGHT;
                cv::Mat ori_img;
                cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
                cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
                cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
                std::vector<cv::Mat> sensor_rgb;
                sensor_rgb.push_back(ori_img_B);
                sensor_rgb.push_back(ori_img_G);
                sensor_rgb.push_back(ori_img_R);
                cv::merge(sensor_rgb, ori_img);

                for(int i = 0; i < results_box.size(); i++)
                {
                    vector<Point> vec;
                    vector<Point2f> sort_vtd(4);
                    vec.clear();
                    for(int j = 0; j < 4; j++)
                    {
                        vec.push_back(results_box[i].vertices[j]);
                    }
                    cv::RotatedRect rect = cv::minAreaRect(vec);
                    cv::Point2f ver[4];
                    rect.points(ver);

                    cv::Mat crop;
                    Utils::warppersp(ori_img, crop, results_box[i], sort_vtd);

                    ocrrec.pre_process(crop);
                    ocrrec.inference();

                    vector<unsigned char> results_rec;
                    ocrrec.post_process(results_rec);
                    Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results_rec,{osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }

                {
                    ScopedTiming st("osd draw", atoi(argv[4]));
                    // cv::putText(osd_frame, "This !!!", {50, 600}, 4, 4, cv::Scalar(255, 255, 255, 255), 4, 8, 0);
                    // cv::rectangle(osd_frame, cv::Rect(osd_width-(200 + 200), osd_height-(700 + 700) , 200, 700), cv::Scalar(255,255, 255, 255), 2, 2, 0);
                    Utils::draw_ocr_det_res(osd_frame, results_box, {osd_height, osd_width}, {SENSOR_WIDTH, SENSOR_HEIGHT});
                }

                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }
            #endif
        }
        #else
        {
            int matsize = SENSOR_WIDTH * SENSOR_HEIGHT;
            cv::Mat ori_img;
            cv::Mat ori_img_R = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr);
            cv::Mat ori_img_G = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 1 * matsize);
            cv::Mat ori_img_B = cv::Mat(SENSOR_HEIGHT, SENSOR_WIDTH, CV_8UC1, vaddr + 2 * matsize);
            std::vector<cv::Mat> sensor_rgb;
            sensor_rgb.push_back(ori_img_B);
            sensor_rgb.push_back(ori_img_G);
            sensor_rgb.push_back(ori_img_R);
            cv::merge(sensor_rgb, ori_img);

            for(int i = 0; i < results_box.size(); i++)
            {
                vector<Point> vec;
                vector<Point2f> sort_vtd(4);
                vec.clear();
                for(int j = 0; j < 4; j++)
                {
                    vec.push_back(results_box[i].vertices[j]);
                }
                cv::RotatedRect rect = cv::minAreaRect(vec);
                cv::Point2f ver[4];
                rect.points(ver);

                cv::Mat crop;
                Utils::warppersp(ori_img, crop, results_box[i], sort_vtd);

                ocrrec.pre_process(crop);
                ocrrec.inference();

                vector<unsigned char> results_rec;
                ocrrec.post_process(results_rec);

                Utils::draw_ocr_text(float(sort_vtd[3].x), float(sort_vtd[3].y),osd_frame,results_rec,{osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }

            {
                ScopedTiming st("osd draw", atoi(argv[4]));
                // cv::putText(osd_frame, "This !!!", {50, 600}, 4, 4, cv::Scalar(255, 255, 255, 255), 4, 8, 0);
                // cv::rectangle(osd_frame, cv::Rect(osd_width-(200 + 200), osd_height-(700 + 700) , 200, 700), cv::Scalar(255,255, 255, 255), 2, 2, 0);
                Utils::draw_ocr_det_res(osd_frame, results_box, {osd_width, osd_height}, {SENSOR_WIDTH, SENSOR_HEIGHT});
            }

        }
        #endif

        {
            ScopedTiming st("osd copy", atoi(argv[4]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            //显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id+3, &vf_info);  //K_VO_OSD0
            // printf("kd_mpi_vo_chn_insert_frame success \n");

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret) {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();


    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

}

void video_proc_ml(config_args args, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL; // osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    Metriclearning ml(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));

    int idx = 0;
    while (!isp_stop)
    {
        ScopedTiming st("total time", atoi(argv[3]));

        {
            ScopedTiming st("read capture", atoi(argv[3]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0, sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }

        {
            ScopedTiming st("isp copy", atoi(argv[3]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3); // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        ml.pre_process();
        ml.inference();

        std::string name = "result_" + std::to_string(idx) + ".bin";
        ml.post_process(name.c_str());

        idx++;

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        {
            ScopedTiming st("osd copy", atoi(argv[3]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            // 显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info); // K_VO_OSD0
            // printf("kd_mpi_vo_chn_insert_frame success \n");

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();

    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
}

void video_proc_seg(config_args args, char *argv[])
{
    vivcap_start();

    k_video_frame_info vf_info;
    void *pic_vaddr = NULL; // osd

    memset(&vf_info, 0, sizeof(vf_info));

    vf_info.v_frame.width = osd_width;
    vf_info.v_frame.height = osd_height;
    vf_info.v_frame.stride[0] = osd_width;
    vf_info.v_frame.pixel_format = PIXEL_FORMAT_ARGB_8888;
    block = vo_insert_frame(&vf_info, &pic_vaddr);

    // alloc memory
    size_t paddr = 0;
    void *vaddr = nullptr;
    size_t size = SENSOR_CHANNEL * SENSOR_HEIGHT * SENSOR_WIDTH;
    int ret = kd_mpi_sys_mmz_alloc_cached(&paddr, &vaddr, "allocate", "anonymous", size);
    if (ret)
    {
        std::cerr << "physical_memory_block::allocate failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }

    Segmentation seg(args, {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, reinterpret_cast<uintptr_t>(vaddr), reinterpret_cast<uintptr_t>(paddr), atoi(argv[3]));

    vector<cls_res> results;

    while (!isp_stop)
    {
        ScopedTiming st("total time", atoi(argv[3]));

        {
            ScopedTiming st("read capture", atoi(argv[3]));
            // VICAP_CHN_ID_1 out rgb888p
            memset(&dump_info, 0, sizeof(k_video_frame_info));
            ret = kd_mpi_vicap_dump_frame(vicap_dev, VICAP_CHN_ID_1, VICAP_DUMP_YUV, &dump_info, 1000);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_frame failed.\n");
                continue;
            }
        }

        {
            ScopedTiming st("isp copy", atoi(argv[3]));
            // 从vivcap中读取一帧图像到dump_info
            auto vbvaddr = kd_mpi_sys_mmap_cached(dump_info.v_frame.phys_addr[0], size);
            memcpy(vaddr, (void *)vbvaddr, SENSOR_HEIGHT * SENSOR_WIDTH * 3); // 这里以后可以去掉，不用copy
            kd_mpi_sys_munmap(vbvaddr, size);
        }

        results.clear();

        seg.pre_process();
        seg.inference();

        cv::Mat pred_color;
        seg.post_process(pred_color);

        cv::Mat osd_frame(osd_height, osd_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));

        {

            #if defined(CONFIG_BOARD_K230D_CANMV)
            {
                ScopedTiming st("osd draw", atoi(argv[3]));
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                cv::Mat tmp_mat_R, tmp_mat_G, tmp_mat_B;
                std::vector<cv::Mat> bgrChannels(3);
                cv::split(pred_color, bgrChannels);

                cv::resize(bgrChannels[2], tmp_mat_R, cv::Size(osd_height, osd_width), cv::INTER_AREA);
                cv::resize(bgrChannels[1], tmp_mat_G, cv::Size(osd_height, osd_width), cv::INTER_AREA);
                cv::resize(bgrChannels[0], tmp_mat_B, cv::Size(osd_height, osd_width), cv::INTER_AREA);

                uint8_t *p_r_addr = reinterpret_cast<uint8_t *>(tmp_mat_R.data);
                uint8_t *p_g_addr = reinterpret_cast<uint8_t *>(tmp_mat_G.data);
                uint8_t *p_b_addr = reinterpret_cast<uint8_t *>(tmp_mat_B.data);

                for (uint32_t hh = 0; hh < osd_height; hh++)
                {
                    for (uint32_t ww = 0; ww < osd_width; ww++)
                    {
                        int new_hh = hh;
                        int new_ww = ww;
                        int osd_channel_index = (new_hh * osd_width + new_ww) * 4;
                        if (p_r_addr[new_hh * osd_width + new_ww] != 0)
                        {
                            int ori_pix_index = hh * osd_width + ww;
                            osd_frame.data[osd_channel_index + 0] = 127;
                            osd_frame.data[osd_channel_index + 1] = p_r_addr[ori_pix_index];
                            osd_frame.data[osd_channel_index + 2] = p_g_addr[ori_pix_index];
                            osd_frame.data[osd_channel_index + 3] = p_b_addr[ori_pix_index];
                        }
                    }
                }
                cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
            }
            #elif defined(CONFIG_BOARD_K230_CANMV_01STUDIO)
            {

                #if defined(STUDIO_HDMI)
                {
                    cv::Mat tmp_mat_R, tmp_mat_G, tmp_mat_B;
                    std::vector<cv::Mat> bgrChannels(3);
                    cv::split(pred_color, bgrChannels);

                    cv::resize(bgrChannels[2], tmp_mat_R, cv::Size(osd_width, osd_height), cv::INTER_AREA);
                    cv::resize(bgrChannels[1], tmp_mat_G, cv::Size(osd_width, osd_height), cv::INTER_AREA);
                    cv::resize(bgrChannels[0], tmp_mat_B, cv::Size(osd_width, osd_height), cv::INTER_AREA);

                    uint8_t *p_r_addr = reinterpret_cast<uint8_t *>(tmp_mat_R.data);
                    uint8_t *p_g_addr = reinterpret_cast<uint8_t *>(tmp_mat_G.data);
                    uint8_t *p_b_addr = reinterpret_cast<uint8_t *>(tmp_mat_B.data);

                    for (uint32_t hh = 0; hh < osd_height; hh++)
                    {
                        for (uint32_t ww = 0; ww < osd_width; ww++)
                        {
                            int new_hh = hh;
                            int new_ww = ww;
                            int osd_channel_index = (new_hh * osd_width + new_ww) * 4;
                            if (p_r_addr[new_hh * osd_width + new_ww] != 0)
                            {
                                int ori_pix_index = hh * osd_width + ww;
                                osd_frame.data[osd_channel_index + 0] = 127;
                                osd_frame.data[osd_channel_index + 1] = p_r_addr[ori_pix_index];
                                osd_frame.data[osd_channel_index + 2] = p_g_addr[ori_pix_index];
                                osd_frame.data[osd_channel_index + 3] = p_b_addr[ori_pix_index];
                            }
                        }
                    }
                }
                #else
                {
                    ScopedTiming st("osd draw", atoi(argv[3]));
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_COUNTERCLOCKWISE);
                    cv::Mat tmp_mat_R, tmp_mat_G, tmp_mat_B;
                    std::vector<cv::Mat> bgrChannels(3);
                    cv::split(pred_color, bgrChannels);

                    cv::resize(bgrChannels[2], tmp_mat_R, cv::Size(osd_height, osd_width), cv::INTER_AREA);
                    cv::resize(bgrChannels[1], tmp_mat_G, cv::Size(osd_height, osd_width), cv::INTER_AREA);
                    cv::resize(bgrChannels[0], tmp_mat_B, cv::Size(osd_height, osd_width), cv::INTER_AREA);

                    uint8_t *p_r_addr = reinterpret_cast<uint8_t *>(tmp_mat_R.data);
                    uint8_t *p_g_addr = reinterpret_cast<uint8_t *>(tmp_mat_G.data);
                    uint8_t *p_b_addr = reinterpret_cast<uint8_t *>(tmp_mat_B.data);

                    for (uint32_t hh = 0; hh < osd_height; hh++)
                    {
                        for (uint32_t ww = 0; ww < osd_width; ww++)
                        {
                            int new_hh = hh;
                            int new_ww = ww;
                            int osd_channel_index = (new_hh * osd_width + new_ww) * 4;
                            if (p_r_addr[new_hh * osd_width + new_ww] != 0)
                            {
                                int ori_pix_index = hh * osd_width + ww;
                                osd_frame.data[osd_channel_index + 0] = 127;
                                osd_frame.data[osd_channel_index + 1] = p_r_addr[ori_pix_index];
                                osd_frame.data[osd_channel_index + 2] = p_g_addr[ori_pix_index];
                                osd_frame.data[osd_channel_index + 3] = p_b_addr[ori_pix_index];
                            }
                        }
                    }
                    cv::rotate(osd_frame, osd_frame, cv::ROTATE_90_CLOCKWISE);
                }
                #endif
            }
            #else
            {
                cv::Mat tmp_mat_R, tmp_mat_G, tmp_mat_B;
                std::vector<cv::Mat> bgrChannels(3);
                cv::split(pred_color, bgrChannels);

                cv::resize(bgrChannels[2], tmp_mat_R, cv::Size(osd_width, osd_height), cv::INTER_AREA);
                cv::resize(bgrChannels[1], tmp_mat_G, cv::Size(osd_width, osd_height), cv::INTER_AREA);
                cv::resize(bgrChannels[0], tmp_mat_B, cv::Size(osd_width, osd_height), cv::INTER_AREA);

                uint8_t *p_r_addr = reinterpret_cast<uint8_t *>(tmp_mat_R.data);
                uint8_t *p_g_addr = reinterpret_cast<uint8_t *>(tmp_mat_G.data);
                uint8_t *p_b_addr = reinterpret_cast<uint8_t *>(tmp_mat_B.data);

                for (uint32_t hh = 0; hh < osd_height; hh++)
                {
                    for (uint32_t ww = 0; ww < osd_width; ww++)
                    {
                        int new_hh = hh;
                        int new_ww = ww;
                        int osd_channel_index = (new_hh * osd_width + new_ww) * 4;
                        if (p_r_addr[new_hh * osd_width + new_ww] != 0)
                        {
                            int ori_pix_index = hh * osd_width + ww;
                            osd_frame.data[osd_channel_index + 0] = 127;
                            osd_frame.data[osd_channel_index + 1] = p_r_addr[ori_pix_index];
                            osd_frame.data[osd_channel_index + 2] = p_g_addr[ori_pix_index];
                            osd_frame.data[osd_channel_index + 3] = p_b_addr[ori_pix_index];
                        }
                    }
                }
            }
            #endif
        }

        {
            ScopedTiming st("osd copy", atoi(argv[3]));
            memcpy(pic_vaddr, osd_frame.data, osd_width * osd_height * 4);
            // 显示通道插入帧
            kd_mpi_vo_chn_insert_frame(osd_id + 3, &vf_info); // K_VO_OSD0
            // printf("kd_mpi_vo_chn_insert_frame success \n");

            ret = kd_mpi_vicap_dump_release(vicap_dev, VICAP_CHN_ID_1, &dump_info);
            if (ret)
            {
                printf("sample_vicap...kd_mpi_vicap_dump_release failed.\n");
            }
        }
    }

    vo_osd_release_block();
    vivcap_stop();

    // free memory
    ret = kd_mpi_sys_mmz_free(paddr, vaddr);
    if (ret)
    {
        std::cerr << "free failed: ret = " << ret << ", errno = " << strerror(errno) << std::endl;
        std::abort();
    }
}

int video_proc(char *argv[])
{
    config_args args;
    string config = argv[1];
    init();
    Utils::parse_args(config, args);
    auto model = modeltype.find(args.model_type);

    if (model == modeltype.end())
    {
        cout << "model_type不匹配" << endl;
        return -1;
    }
    int modelnum = model->second;
    if (modelnum < 10)
    {
        video_proc_ob_det(args, argv);
        return -1;
    }
    else if (modelnum < 20)
    {
        video_proc_cls(args, argv);
        return -1;
    }
    else if (modelnum < 30)
    {
        video_proc_seg(args,argv);
        return -1;
    }
    else if (modelnum < 40)
    {
        video_proc_ocr_det(args, argv);
        return -1;
    }
    else if (modelnum < 50)
    {
        std::cout << "不支持此种命令,请按 q 退出命令" << std::endl;
        return -1;
    }
    else if (modelnum < 60)
    {
        std::cout << "不支持此种命令,请按 q 退出命令" << std::endl;
        return -1;
    }
    else if (modelnum < 70)
    {
        video_proc_mlcls(args, argv);
        return -1;
    }
    else if (modelnum < 80)
    {
        video_proc_ml(args, argv);
        return -1;
    }
    else
    {
        std::cout << "不支持此种命令,请按 q 退出命令" << std::endl;
        return -1;
    }
}

int video_proc_two_works(char *argv[])
{
    config_args args1;
    string config1 = argv[1];
    init();
    Utils::parse_args(config1, args1);
    auto model1 = modeltype.find(args1.model_type);

    config_args args2;
    string config2 = argv[2];
    Utils::parse_args(config2, args2);
    auto model2 = modeltype.find(args2.model_type);

    if (model1 == modeltype.end() || model2 == modeltype.end())
    {
        cout << "model_type不匹配" << endl;
        return -1;
    }

    int modelnum1 = model1->second;
    int modelnum2 = model2->second;

    if ((modelnum1 < 40 && modelnum1 >= 30) && (modelnum2 >= 40 && modelnum2 < 50))
    {
        video_proc_ocr_det_rec(args1, args2, argv);
        return -1;
    }
    else
    {
        std::cout << "不支持此种命令,请按 q 退出命令" << std::endl;
        return -1;
    }
}

int image_proc(char *argv[])
{
    config_args args;
    string config = argv[1];
    init();
    Utils::parse_args(config, args);

    auto model = modeltype.find(args.model_type);
    
    if (model == modeltype.end())
    {
        cout << "model_type不匹配" << endl;
        return -1;
    }
    int modelnum = model->second;
    
    if (modelnum < 10)
    {
        image_proc_ob_det(args, argv);
        return -1;
    }
    else if (modelnum < 20)
    {
        image_proc_cls(args, argv);
        return -1;

    }
    else if (modelnum < 30)
    {
        image_proc_seg(args, argv);
        return -1;
    }
    else if (modelnum < 40)
    {
        image_proc_ocr_det(args, argv);
        return -1;
    }
    else if (modelnum < 50)
    {
        image_proc_ocr_rec(args, argv);
        return -1;
    }
    else if (modelnum < 60)
    {
        image_proc_anomaly(args, argv);
        return -1;
    }
    else if (modelnum < 70)
    {
        image_proc_mlcls(args, argv);
        return -1;
    }
    else if (modelnum < 80)
    {
        image_proc_ml(args, argv);
        return -1;
    }    
    else
    {
        return -1;
    }

}

int image_proc_two_works(char *argv[])
{
    config_args args1;
    string config1 = argv[1];
    init();
    Utils::parse_args(config1, args1);
    auto model1 = modeltype.find(args1.model_type);

    config_args args2;
    string config2 = argv[2];
    Utils::parse_args(config2, args2);
    auto model2 = modeltype.find(args2.model_type);

    if (model1 == modeltype.end() || model2 == modeltype.end())
    {
        cout << "model_type不匹配" << endl;
        return -1;
    }

    int modelnum1 = model1->second;
    int modelnum2 = model2->second;


    if ((modelnum1 < 40 && modelnum1 >= 30) && (modelnum2 >= 40 && modelnum2 < 50))
    {
        image_proc_ocr_det_rec(args1, args2, argv);
        return -1;
    }
    else
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc < 4 or argc > 5)
    {
        print_usage();
        return -1;
    }

    if (argc == 4)
    {   
        //video
        if (strcmp(argv[2], "None") == 0)
        {
            std::thread thread_isp(video_proc, argv);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
            return -1;
        }
        //image
        else
        {
            image_proc(argv);
            
            return -1;
        }
    }
    else
    {
        //video
        if (strcmp(argv[3], "None") == 0)
        {
            std::thread thread_isp(video_proc_two_works, argv);
            while (getchar() != 'q')
            {
                usleep(10000);
            }

            isp_stop = true;
            thread_isp.join();
            return -1;
        }
        //image
        else
        {
            image_proc_two_works(argv);
            return -1;
        }
    }
    return 0;
}
