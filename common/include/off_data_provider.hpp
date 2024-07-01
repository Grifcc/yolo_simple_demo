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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_DATA_PROVIDER_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_DATA_PROVIDER_HPP_
#include <queue>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "runner.hpp"
#include "off_runner.hpp"
#include "data_provider.hpp"
#include "command_option.hpp"
#include "cnrt.h" // NOLINT

template <class Dtype, template <class...> class Qtype>
class OffDataProvider: public DataProvider<Dtype, Qtype>{
  public:
  explicit OffDataProvider(
      const std::string& meanvalue,
      const std::string& stdvalue,
      const std::queue<std::string>& images): DataProvider<Dtype, Qtype>(meanvalue, stdvalue, images) {}
  virtual ~OffDataProvider() {
    delete [] reinterpret_cast<float*>(cpuData_[0]);
    delete cpuData_;
    delete [] reinterpret_cast<float*>(cpuTempData_[0]);
    delete cpuTempData_;
    delete [] reinterpret_cast<float*>(cpuTrans_[0]);
    delete cpuTrans_;
    delete [] reinterpret_cast<char*>(firstConvData_[0]);
    delete firstConvData_;
    if (FLAGS_async_copy) {
      for (auto queue : asyncCopyQueueVector_) {
        cnrtDestroyQueue(queue);
      }
    }
  }
  inline void pushAsyncCopyQueueVector(cnrtQueue_t queue) { asyncCopyQueueVector_.push_back(queue); } //NOLINT
  virtual void runParallel();

  protected:
  void allocateMemory(int queueLength);
  void init();
  void readInputData();
  void prepareInput();
  void transData();

  private:
  Dtype* cpuData_;
  Dtype* cpuTrans_;
  Dtype* cpuTempData_;
  Dtype* firstConvData_;
  std::vector<cnrtQueue_t> asyncCopyQueueVector_;
  cnrtDataType_t* input_data_type;
  int dim_order[4];
  int dim_shape[4];
  int inputDimValue[4];
  int inputDimStride[4];
  int input_count;
  std::vector<void *> temp_ptrs;
};
#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_OFF_DATA_PROVIDER_HPP_
