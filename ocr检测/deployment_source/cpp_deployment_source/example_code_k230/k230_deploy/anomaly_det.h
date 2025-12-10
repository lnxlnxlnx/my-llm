/* Copyright (c) 2022, Canaan Bright Sight Co., Ltd
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

#ifndef _ANOMALY_DET_H_
#define _ANOMALY_DET_H_

#include "ai_base.h"
#include "utils.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;

// ANOMALY_DET
class AnomalyDet:public AIBase
{
public:
	/**
     * @brief AnomalyDet构造函数，加载kmodel,并初始化kmodel输入、输出和异常检测阈值
     * @param kmodel_file kmodel文件路径
     * @param obj_thresh  异常检测阈值
     * @param debug_mode  0（不调试）、 1（只显示时间）、2（显示所有打印信息）
     * @return None
     */
    // AnomalyDet(const char *kmodel_file, float obj_thresh, const int debug_mode = 1);
	AnomalyDet(config_args args, const int debug_mode = 1);
	~AnomalyDet(){}
	/**
     * @brief 图片预处理，（ai2d for image）
     * @param ori_img 原始图片
     * @return None
     */
    void pre_process(cv::Mat ori_img);

	/**
     * @brief kmodel推理
     * @return None
     */
    void inference();

	/**
     * @brief kmodel推理结果后处理
     * @param final_score 异常检测得分
     * @return None
     */
    void post_process(vector<anomaly_res> &results);

private:
	// Framesize frame_size;	
	Eigen::MatrixXf cdist_eigen_matmul(const Eigen::MatrixXf& x1, const Eigen::MatrixXf& x2);
	Eigen::MatrixXf softmax(Eigen::MatrixXf x);
	std::pair<Eigen::VectorXf, Eigen::VectorXi> find_top_k_values(Eigen::MatrixXf mat, int k);
	std::pair<Eigen::VectorXf, Eigen::VectorXi> nearest_neighbors(Eigen::MatrixXf embedding, Eigen::MatrixXf memory_bank, int k);
	float compute_anomaly_score(Eigen::VectorXf patch_score, Eigen::VectorXi locations, Eigen::MatrixXf embedding, Eigen::MatrixXf memory_bank, int k);
	Eigen::Tensor<float, 3> nearest_neighbor_interpolation(Eigen::Tensor<float, 3> input_image, std::pair<int, int> target_size);
	Eigen::Tensor<float, 3> avgPool(const Eigen::Tensor<float, 3>& input, int poolHeight, int poolWidth, int strideH, int strideW, int padH, int padW);

	std::unique_ptr<ai2d_builder> ai2d_builder_; // ai2d构建器
    runtime_tensor ai2d_in_tensor_;              // ai2d输入tensor
    runtime_tensor ai2d_out_tensor_;             // ai2d输出tensor

    float obj_thresh_; // 异常检测阈值
    vector<int> memory_shape_;
    string labels_;
    
};

#endif