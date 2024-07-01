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

#pragma once

#include "yolov5_processor.hpp"
#include "threadPool.h"
#include "simple_interface.hpp"

template<typename Dtype, template <typename> class Qtype>
class YoloV5OffPostProcessor: public YoloV5Processor<Dtype, Qtype> {
 public:
  YoloV5OffPostProcessor() {}
  ~YoloV5OffPostProcessor() {
    delete [] reinterpret_cast<float*>(outCpuPtrs_[0]);
    delete outCpuPtrs_;
    delete [] reinterpret_cast<float*>(outTrans_[0]);
    delete outTrans_;
    delete [] reinterpret_cast<char*>(outTempCpuPtrs_[0]);
    delete outTempCpuPtrs_;
  }
  virtual void runParallel();

 private:
  std::vector<std::vector<std::vector<float>>> getResults(int outShape);

  void getImages(std::vector<cv::Mat> *imgs,
                 std::vector<std::string> *img_names,
                 const std::vector<std::pair<std::string, cv::Mat>> &origin_img);
 private:
  Dtype* outCpuPtrs_;
  Dtype* outTrans_;
  Dtype* outTempCpuPtrs_;
};
