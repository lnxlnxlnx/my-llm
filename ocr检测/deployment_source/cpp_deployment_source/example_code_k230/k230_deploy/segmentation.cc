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

#include "segmentation.h"

Segmentation::Segmentation(config_args args, const int debug_mode)
:thresh(args.obj_thresh),labels(args.labels), AIBase(args.kmodel_path.c_str(),"Segmentation", debug_mode)
{
    num_class = labels.size();
    ai2d_out_tensor_ = this -> get_input_tensor(0);
}

Segmentation::Segmentation(config_args args, FrameCHWSize isp_shape, uintptr_t vaddr, uintptr_t paddr,const int debug_mode)
:thresh(args.obj_thresh),labels(args.labels), AIBase(args.kmodel_path.c_str(),"Segmentation", debug_mode)
{
    num_class = labels.size();

    vaddr_ = vaddr;

    isp_shape_ = isp_shape;
    dims_t in_shape{1, isp_shape.channel, isp_shape.height, isp_shape.width};
    // int isp_size = isp_shape.channel * isp_shape.height * isp_shape.width;

    ai2d_in_tensor_ = hrt::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("create ai2d input tensor failed");

    ai2d_out_tensor_ = this -> get_input_tensor(0);

    Utils::resize(ai2d_builder_, ai2d_in_tensor_, ai2d_out_tensor_);
}

Segmentation::~Segmentation()
{

}

void Segmentation::pre_process(cv::Mat ori_img)
{
    ScopedTiming st(model_name_ + " pre_process image", debug_mode_);
    std::vector<uint8_t> chw_vec;
    Utils::bgr2rgb_and_hwc2chw(ori_img, chw_vec);
    Utils::resize({ori_img.channels(), ori_img.rows, ori_img.cols}, chw_vec, ai2d_out_tensor_);
}

void Segmentation::pre_process()
{
    ScopedTiming st(model_name_ + " pre_process video", debug_mode_);
    size_t isp_size = isp_shape_.channel * isp_shape_.height * isp_shape_.width;
    auto buf = ai2d_in_tensor_.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    memcpy(reinterpret_cast<char *>(buf.data()), (void *)vaddr_, isp_size);
    hrt::sync(ai2d_in_tensor_, sync_op_t::sync_write_back, true).expect("sync write_back failed");
    ai2d_builder_->invoke(ai2d_in_tensor_, ai2d_out_tensor_).expect("error occurred in ai2d running");
}

void Segmentation::inference()
{
    this->run();
    this->get_output();
}


void Segmentation::post_process(cv::Mat& images_pred_color)
{
    ScopedTiming st(model_name_ + " post_process", debug_mode_);
    output = p_outputs_[0];
    // cv::resize(in_image, resized_image, cv::Size(input_width, input_height));
    images_pred_color = cv::Mat::zeros(input_shapes_[0][2], input_shapes_[0][3], CV_8UC3);

    for (int y = 0; y < input_shapes_[0][2]; ++y)
    {
        for (int x = 0; x < input_shapes_[0][3]; ++x)
        {
            float s = 0.0;
            int loc = num_class * (x + y * input_shapes_[0][3]);
            for (int c = 0; c < num_class; c++)
                s += exp(output[loc + c]);
            vector<float> scores(num_class);
            scores.clear();
            for (int c = 0; c < num_class; c++)
            {
                output[loc + c] = output[loc + c] / s;
                scores.push_back(output[loc + c]);
            }
            cv::Vec3b& color = images_pred_color.at<cv::Vec3b>(cv::Point(x, y));
            color[0] = 0;
            color[1] = 0;
            color[2] = 0;
            float score0 = scores[0];
            for (int i = 1; i < num_class; ++i)
            {
                if (scores[i] > score0)
                {
                    score0 = scores[i];
                    cv::Vec3b& color = images_pred_color.at<cv::Vec3b>(cv::Point(x, y));
                    color[0] = max(255 - i * 60, 0);
                    color[1] = min(i * 80, 255);
                    color[2] = max(255 - (num_class - i) * 100, 255);
                }
            }
        }
    }
    resize(images_pred_color, images_pred_color, cv::Size(input_shapes_[0][2], input_shapes_[0][3]));
}