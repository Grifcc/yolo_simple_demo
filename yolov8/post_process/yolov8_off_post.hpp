/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "yolov8_processor.hpp"
#include "threadPool.h"
#include "simple_interface.hpp"

template<typename Dtype, template <typename> class Qtype>
class YoloV8OffPostProcessor: public YoloV8Processor <Dtype, Qtype> {
 public:
  YoloV8OffPostProcessor() {}
  ~YoloV8OffPostProcessor() {
    // delete [] reinterpret_cast<float*>(outCpuPtrs_[0]);
    // delete outCpuPtrs_;
    // delete [] reinterpret_cast<float*>(outTrans_[0]);
    // delete outTrans_;
    // delete [] reinterpret_cast<char*>(outTempCpuPtrs_[0]);
    // delete outTempCpuPtrs_;
    for (auto e : outCpuPtrs_)
    {
      delete e;
    }
    for (auto e : outTrans_)
    {
      delete e;
    }
    for (auto e : outTempCpuPtrs_)
    {
      delete e;
    }
  }
  virtual void runParallel();
  
 private:

  std::vector<std::vector<std::vector<float>>> getResults(int outShape);

  std::vector<std::vector<std::vector<float>>> dist2bbox(float *input,
              const std::vector<std::vector<float>> anchors,
              const std::vector<float> strides,
              const int grid_h,
              const int grid_w,
              const int batch);

  std::vector<std::vector<std::vector<float>>> process(float *cls_score,
                                                    std::vector<std::vector<std::vector<float>>> dbox, 
                                                    const int cls_grid_h,
                                                    const int box_grid_h,
                                                    const int grid_w,
                                                    const int batch,
                                                    const float conf_thres = 0.45);
  
    
  std::vector<std::vector<std::vector<float>>> nonMaximumSuppression(std::vector<std::vector<std::vector<float>>> boxes, 
                                                    const float iou_thres, 
                                                    const int classnum);

  void make_anchors(std::vector<std::vector<float>> &anchors,
              std::vector<float> &strides,
              const int grid_h,
              const int grid_w,
              const float grid_cell_offset);
              
  void getImages(std::vector<cv::Mat> *imgs,
                 std::vector<std::string> *img_names,
                 const std::vector<std::pair<std::string, cv::Mat>> &origin_img);

      
 private:
  const float conf_thres = 0.45;
  const float iou_thres = 0.6;
  const int size_img = 640;
  const int nc = 5;
  const int reg_max = 16;
  const int featshapes[3] = {80,40,20};
  // const int angle_cls = 180;
  const int no = nc + reg_max * 4;

  std::vector<std::vector<float>> anchors;
  std::vector<float> strides; 

  std::vector<float *> outCpuPtrs_;
  std::vector<float *> outTrans_;
  std::vector<char *> outTempCpuPtrs_;
};
