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

#include "gfl_det.h"

GFLDet::GFLDet(config_args args, const int debug_mode)
:ob_det_thresh(args.obj_thresh),ob_nms_thresh(args.nms_thresh),labels(args.labels), AIBase(args.kmodel_path.c_str(),"GFLDet", debug_mode)
{
    nms_option = args.nms_option;
    num_class = labels.size();
    reg_max = REG_MAX;

    memcpy(this->strides, args.strides, sizeof(args.strides));

    input_width = input_shapes_[0][3];
    input_height = input_shapes_[0][2];

    generate_grid_center_priors();

    ai2d_out_tensor_ = this -> get_input_tensor(0);
}

GFLDet::GFLDet(config_args args, FrameCHWSize isp_shape, uintptr_t vaddr, uintptr_t paddr,const int debug_mode)
:ob_det_thresh(args.obj_thresh),ob_nms_thresh(args.nms_thresh),labels(args.labels), AIBase(args.kmodel_path.c_str(),"GFLDet", debug_mode)
{
    nms_option = args.nms_option;
    num_class = labels.size();
    reg_max = REG_MAX;

    memcpy(this->strides, args.strides, sizeof(args.strides));

    input_width = input_shapes_[0][3];
    input_height = input_shapes_[0][2];

    generate_grid_center_priors();

    vaddr_ = vaddr;

    isp_shape_ = isp_shape;
    dims_t in_shape{1, isp_shape.channel, isp_shape.height, isp_shape.width};

    ai2d_in_tensor_ = hrt::create(typecode_t::dt_uint8, in_shape, hrt::pool_shared).expect("create ai2d input tensor failed");

    ai2d_out_tensor_ = this -> get_input_tensor(0);

    Utils::padding_resize(isp_shape, {input_shapes_[0][3], input_shapes_[0][2]}, ai2d_builder_, ai2d_in_tensor_, ai2d_out_tensor_, cv::Scalar(114, 114, 114));
}

GFLDet::~GFLDet()
{

}

void GFLDet::pre_process(cv::Mat ori_img)
{
    ScopedTiming st(model_name_ + " pre_process image", debug_mode_);
    std::vector<uint8_t> chw_vec;
    Utils::bgr2rgb_and_hwc2chw(ori_img, chw_vec);
    Utils::padding_resize({ori_img.channels(), ori_img.rows, ori_img.cols}, chw_vec, {input_shapes_[0][3], input_shapes_[0][2]}, ai2d_out_tensor_, cv::Scalar(114, 114, 114));
}

void GFLDet::pre_process()
{
    ScopedTiming st(model_name_ + " pre_process video", debug_mode_);
    size_t isp_size = isp_shape_.channel * isp_shape_.height * isp_shape_.width;
    auto buf = ai2d_in_tensor_.impl()->to_host().unwrap()->buffer().as_host().unwrap().map(map_access_::map_write).unwrap().buffer();
    memcpy(reinterpret_cast<char *>(buf.data()), (void *)vaddr_, isp_size);
    hrt::sync(ai2d_in_tensor_, sync_op_t::sync_write_back, true).expect("sync write_back failed");
    ai2d_builder_->invoke(ai2d_in_tensor_, ai2d_out_tensor_).expect("error occurred in ai2d running");
}

void GFLDet::inference()
{
    this->run();
    this->get_output();
}

void GFLDet::post_process(FrameSize frame_size, vector<ob_det_res> &results)
{
    ScopedTiming st(model_name_ + " post_process", debug_mode_);
    output_0 = p_outputs_[0];
    output_1 = p_outputs_[1];
    output_2 = p_outputs_[2];

    if(nms_option)
    {
        vector<ob_det_res> b0, b1, b2;

        decode_infer(output_0, center_priors[0], b0, frame_size);
        decode_infer(output_1, center_priors[1], b1, frame_size);
        decode_infer(output_2, center_priors[2], b2, frame_size);

        results.insert(results.begin(), b0.begin(), b0.end());
        results.insert(results.begin(), b1.begin(), b1.end());
        results.insert(results.begin(), b2.begin(), b2.end());

        nms(results);
    }
    else
    {
        vector<vector<ob_det_res>> b0, b1, b2;
        for (int i = 0; i < num_class; i++)
        {
            b0.push_back(vector<ob_det_res>());//不断往v2d里加行 
            b1.push_back(vector<ob_det_res>());//不断往v2d里加行 
            b2.push_back(vector<ob_det_res>());//不断往v2d里加行 
        }

        decode_infer_class(output_0, center_priors[0], b0, frame_size);
        decode_infer_class(output_1, center_priors[1], b1, frame_size);
        decode_infer_class(output_2, center_priors[2], b2, frame_size);

        for(int i = 0; i < num_class; i++)
        {
            b0[i].insert(b0[i].begin(), b1[i].begin(), b1[i].end());
            b0[i].insert(b0[i].begin(), b2[i].begin(), b2[i].end());
            nms(b0[i]);
            results.insert(results.begin(), b0[i].begin(), b0[i].end());
        }
    }

}

void GFLDet::generate_grid_center_priors()
{
    for (int i = 0; i < STAGE_NUM; i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++)
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors[i].push_back(ct);
            }
        
    }
}

float GFLDet::fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

float GFLDet::sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
    //return 1.0f / (1.0f + exp(-x));
}

void GFLDet::decode_infer(float* pred, std::vector<CenterPrior>& center_priors,  std::vector<ob_det_res>& results, FrameSize frame_size)
{
    float ratiow = (float)input_width / frame_size.width;
    float ratioh = (float)input_height / frame_size.height;
    float gain = ratiow < ratioh ? ratiow : ratioh;
    const int num_points = center_priors.size();
    const int num_channels = num_class + (reg_max + 1) * 4;
    for (int idx = 0; idx < num_points; idx++)
    {
        int ct_x = center_priors[idx].x;
        int ct_y = center_priors[idx].y;
        int stride = center_priors[idx].stride;
        float score = 0;
        int cur_label = 0;

        for (int label = 0; label < num_class; label++)
        {
            
            float sig_score = sigmoid(pred[idx * num_channels + label]);
            if (sig_score > score)
            {
                score = sig_score;
                cur_label = label;
            }
        }

        if (score > ob_det_thresh)
        {
            const float* bbox_pred = pred + idx * num_channels + num_class;
            results.push_back(disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride, reg_max, input_height, input_width, ratiow, ratioh, gain, frame_size));
        }
    }
}

void GFLDet::decode_infer_class(float* pred, std::vector<CenterPrior>& center_priors,  std::vector<std::vector<ob_det_res>>& results, FrameSize frame_size)
{
    float ratiow = (float)input_width / frame_size.width;
    float ratioh = (float)input_height / frame_size.height;
    float gain = ratiow < ratioh ? ratiow : ratioh;
    const int num_points = center_priors.size();
    const int num_channels = num_class + (reg_max + 1) * 4;
    for (int idx = 0; idx < num_points; idx++)
    {
        int ct_x = center_priors[idx].x;
        int ct_y = center_priors[idx].y;
        int stride = center_priors[idx].stride;
        float score = 0;
        int cur_label = 0;

        for (int label = 0; label < num_class; label++)
        {
            
            float sig_score = sigmoid(pred[idx * num_channels + label]);
            if (sig_score > score)
            {
                score = sig_score;
                cur_label = label;
            }
        }

        if (score > ob_det_thresh)
        {
            const float* bbox_pred = pred + idx * num_channels + num_class;
            ob_det_res tmp_res = disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride, reg_max, input_height, input_width, ratiow, ratioh, gain, frame_size);
            results[tmp_res.label_index].push_back(tmp_res);
        }
    }
}

ob_det_res GFLDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride, int reg_max, int input_height,int input_width,float ratiow, float ratioh, float gain,FrameSize frame_size)
{
    float ct_x = x * stride;
    float ct_y = y * stride;

    ct_x -= ((input_width - frame_size.width * gain) / 2);
    ct_y -= ((input_height - frame_size.height * gain) / 2);
    ct_x /= gain;
    ct_y /= gain;

    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++)
            dis += j * dis_after_sm[j];
        
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0] /gain, .0f);
    float ymin = (std::max)(ct_y - dis_pred[1] /gain, .0f);
    float xmax = (std::min)(ct_x + dis_pred[2] / gain, (float)frame_size.width);
    float ymax = (std::min)(ct_y + dis_pred[3] / gain, (float)frame_size.height);

    return ob_det_res{ xmin, ymin, xmax, ymax, score, label, labels[label] };
}

template<typename _Tp>
int GFLDet::activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        //dst[i] = exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

void GFLDet::nms(std::vector<ob_det_res>& input_boxes)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](ob_det_res a, ob_det_res b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    
    for (int i = 0; i < int(input_boxes.size()); ++i)
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= ob_nms_thresh)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
                j++;
        }
}

