/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_CLAS_DATA_PROVIDER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_CLAS_DATA_PROVIDER_HPP_
#include <queue>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "runner.hpp"
#include "off_data_provider.hpp"

template <class Dtype, template <class...> class Qtype>
class OffClasDataProvider: public OffDataProvider<Dtype, Qtype>{
  public:
  explicit OffClasDataProvider(
      const std::string& meanvalue,
      const std::string& stdvalue,
      const std::queue<std::string>& images,
      const std::vector<int>& resizes_): OffDataProvider<Dtype, Qtype>(meanvalue, stdvalue, images) {
    if (2 == resizes_.size()) {
      in_resizes_.push_back(resizes_[0]);
      in_resizes_.push_back(resizes_[1]);
    } else if (1 == resizes_.size()) {
      in_resizes_.push_back(resizes_[0]);
    }
  }

  virtual ~OffClasDataProvider() {
    in_resizes_.clear();
  }
  virtual void Preprocess(const std::vector<std::pair<std::string, cv::Mat>>& srcImages,
                          std::vector<std::vector<cv::Mat> >* dstImages);

  private:
  std::vector<int> in_resizes_;
};
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_CLAS_DATA_PROVIDER_HPP_
