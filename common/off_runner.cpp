/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include <algorithm>
#include <atomic>
#include <condition_variable> // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include "include/runner.hpp"
#include "include/off_runner.hpp"
#include "include/command_option.hpp"
#include "include/common_functions.hpp"
#include "include/runner_strategy.hpp"

template<class Dtype, template <class...> class Qtype>
OffRunner<Dtype, Qtype>::OffRunner(const cnrtRuntimeContext_t rt_ctx,
                                   const int& id) {
  this->rt_ctx_ = rt_ctx;
  this->threadId_ = id;
  this->runTime_ = 0;
  this->simple_flag_ = true;

  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_FUNCTION,
                            reinterpret_cast<void **>(&this->function_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_MODEL_PARALLEL,
                            reinterpret_cast<void **>(&this->Parallel_));
  cnrtGetRuntimeContextInfo(rt_ctx, CNRT_RT_CTX_DEV_ORDINAL,
                            reinterpret_cast<void **>(&this->deviceId_));

  getIODataDesc();
  runnerStrategy_ = new SimpleStrategy<Dtype, Qtype>();
}

template<class Dtype, template <class...> class Qtype>
OffRunner<Dtype, Qtype>::OffRunner(const std::string& offlinemodel,
                                   const int& id,
                                   const int& parallel,
                                   const int& deviceId,
                                   const int& devicesize) {
  this->threadId_ = id;
  this->deviceId_ = deviceId;
  this->deviceSize_ = devicesize;
  this->runTime_ = 0;
  this->simple_flag_ = false;
  setDeviceId(deviceId);
  loadOfflinemodel(offlinemodel);
  getIODataDesc();
  runnerStrategy_ = new FlexibleStrategy<Dtype, Qtype>();
}

template<class Dtype, template <class...> class Qtype>
OffRunner<Dtype, Qtype>::~OffRunner() {
  setDeviceId(this->deviceId_);

  delete runnerStrategy_;
}


// get function's I/O DataDesc
template<class Dtype, template <class...> class Qtype>
void OffRunner<Dtype, Qtype>::getIODataDesc() {
  cnrtDataType_t* input_data_type = nullptr;
  CNRT_CHECK(cnrtGetInputDataType(&input_data_type, &(this->inBlobNum_),
                                  this->function()));
  cnrtDataType_t* output_data_type = nullptr;
  CNRT_CHECK(cnrtGetOutputDataType(&output_data_type, &(this->outBlobNum_),
                                  this->function()));

  LOG(INFO) << "input blob num is " << this->inBlobNum_;
  auto shape = std::make_shared<int *>();
  for (int i = 0; i < this->inBlobNum_; i++) {
    int dimNum = 4;
    cnrtGetInputDataShape(shape.get(), &dimNum, i, this->function_);  // NHWC
    if (i == 0) {
      this->inNum_ = shape.get()[0][0];
      this->inChannel_ = (input_data_type[i] == CNRT_UINT8) ?
                         (FLAGS_input_format > 0 ? shape.get()[0][3] : (shape.get()[0][3] - 1)) :
                         shape.get()[0][3];
      this->inHeight_ = shape.get()[0][1];
      this->inWidth_ = shape.get()[0][2];
    }
    LOG(INFO) << "shape " << shape.get()[0][0];
    LOG(INFO) << "shape " << this->inChannel_;
    LOG(INFO) << "shape " << shape.get()[0][1];
    LOG(INFO) << "shape " << shape.get()[0][2];
    this->inCount_ = this->inNum_ * this->inChannel_ * this->inHeight_ * this->inWidth_;
    free(shape.get()[0]);  // cnrtGetInputDataShape malloc for shape which need free outside.
  }

  for (int i = 0; i < this->outBlobNum_; i++) {
    int dimNum = 4;
    cnrtGetOutputDataShape(shape.get(), &dimNum, i, this->function_);  // NHWC
    this->outNum_ =  shape.get()[0][0];
    this->outChannel_ = shape.get()[0][3];
    this->outHeight_  = shape.get()[0][1];
    this->outWidth_ = shape.get()[0][2];
    this->outCount_ = this->outNum_ * this->outChannel_ * this->outHeight_ * this->outWidth_;
    this->outallcounts_.push_back(this->outCount_);
    LOG(INFO) << "output shape " << this->outNum_;
    LOG(INFO) << "output shape " << this->outChannel_;
    LOG(INFO) << "output shape " << this->outHeight_;
    LOG(INFO) << "output shape " << this->outWidth_;
    free(shape.get()[0]);  // cnrtGetOutputDataShape malloc for shape which need free outside.
  }
}

// load model and get function
template<class Dtype, template <class...> class Qtype>
void OffRunner<Dtype, Qtype>::loadOfflinemodel(const std::string& offlinemodel) {
  LOG(INFO) << "load file: " << offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());
  CHECK_LE(FLAGS_apiversion, 2) << "The version number should be 1 or 2";
  CHECK_GE(FLAGS_apiversion, 1) << "The version number should be 1 or 2";
  if (FLAGS_functype != "1H8" && FLAGS_functype != "1H16") {
    LOG(FATAL) << "The value of functype can only be set to 1H8 or 1H16";
  }
  const std::string name = "subnet0";
  cnrtCreateFunction(&function_);
  cnrtExtractFunction(&function_, model_, name.c_str());
}

template<class Dtype, template <class...> class Qtype>
void OffRunner<Dtype, Qtype>::runParallel() {
  runnerStrategy_->runParallel(this);
}

INSTANTIATE_OFF_CLASS(OffRunner);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
