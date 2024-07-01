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
#include "include/handler.hpp"

void OfflineDescripter::getIODataDesc() {
  cnrtDataType_t* input_data_type = nullptr;
  int inBlobNum, outBlobNum;
  CNRT_CHECK(cnrtGetInputDataType(&input_data_type, &inBlobNum, this->function_));
  // XXX: need to consider futher about first_conv judgement
  is_first_conv_ = input_data_type[0] == CNRT_UINT8;
  cnrtDataType_t* output_data_type = nullptr;
  CNRT_CHECK(cnrtGetOutputDataType(&output_data_type, &outBlobNum, this->function_));

  LOG(INFO) << "input blob num is " << inBlobNum;
  LOG(INFO) << "outpu blob num is " << outBlobNum;
  shape_descripter_->setShape(inBlobNum, outBlobNum);

  auto shape = std::make_shared<int *>();
  for (int i = 0; i < inBlobNum; i++) {
    int dimNum = 4;
    cnrtGetInputDataShape(shape.get(), &dimNum, i, this->function_);  // NHWC
    shape_descripter_->setInputShapes(shape, i, is_first_conv_);
    // cnrtGetInputDataShape malloc for shape which need free outside.
    free(shape.get()[0]);
  }

  for (int i = 0; i < outBlobNum; i++) {
    int dimNum = 4;
    cnrtGetOutputDataShape(shape.get(), &dimNum, i, this->function_);  // NHWC
    shape_descripter_->setOutputShapes(shape, i);
    // cnrtGetOutputDataShape malloc for shape which need free outside.
    free(shape.get()[0]);
  }

  // set input/output datatype
  CNRT_CHECK(cnrtGetInputDataType(&input_data_type_, &inBlobNum, this->function_));
  CNRT_CHECK(cnrtGetOutputDataType(&output_data_type_, &outBlobNum, this->function_));
}

void OfflineDescripter::loadOfflinemodel(const std::string& offlinemodel,
                                       const bool& channel_dup,
                                       const int threads) {
  LOG(INFO)<< "load file: " << offlinemodel.c_str();
  cnrtLoadModel(&model_, offlinemodel.c_str());

  const std::string name = "subnet0";
  cnrtCreateFunction(&function_);
  cnrtExtractFunction(&function_, model_, name.c_str());

  for (auto device : deviceIds_) {
    LOG(INFO)<< "Init runtime context for device" << device;
    cnrtRuntimeContext_t ctx;
    prepareRuntimeContext(&ctx, function_, device, channel_dup);
    std::vector<cnrtRuntimeContext_t> rctxs;
    rctxs.push_back(ctx);
    for (int i = 1; i < threads; i++) {
      cnrtRuntimeContext_t tmp_ctx;
      cnrtForkRuntimeContext(&tmp_ctx, ctx, NULL);
      rctxs.push_back(tmp_ctx);
    }
    dev_runtime_contexts_.push_back(rctxs);
  }

  getIODataDesc();
  for (auto device : deviceIds_) {
    allocateMemory(FLAGS_fifosize * data_provider_num_, device);
  }
}

void OfflineDescripter::prepareRuntimeContext(cnrtRuntimeContext_t *ctx,
                                            cnrtFunction_t function,
                                            int deviceId,
                                            const bool& channel_dup) {
  cnrtRuntimeContext_t rt_ctx;
  // cnrtRet_t ret;

  if (cnrtCreateRuntimeContext(&rt_ctx, function, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to create runtime context";
  }

  // set device ordinal. if not set, a random device will be used
  if (cnrtSetRuntimeContextDeviceId(rt_ctx, deviceId) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to set runtime context";
  }

  /* set channel is optional. you could use CNRT_CHANNEL_TYPE_DUPLICATE if you want to */
  /* trade space for performance when multithreading */

  // Instantiate the runtime context on actual MLU device
  // All cnrtSetRuntimeContext* interfaces must be caller prior to cnrtInitRuntimeContext
  if (cnrtInitRuntimeContext(rt_ctx, nullptr) != CNRT_RET_SUCCESS) {
    LOG(FATAL)<< "Failed to initialize runtime context";
  }

  *ctx = rt_ctx;
}

void OfflineDescripter:: destroyRuntimeContext() {
  for (auto ctxs : dev_runtime_contexts_) {
    for (auto ctx : ctxs)
      cnrtDestroyRuntimeContext(ctx);
  }
  cnrtUnloadModel(model_);
}

void OfflineDescripter::allocateMemory(int queueLength, int deviceId) {
  setDeviceId(deviceId);
  int inputNum = this->inputNum();
  int outputNum = this->outputNum();
  for (int i = 0; i < queueLength; i++) {
    // this queue is for async copy
    auto queue = std::make_shared<cnrtQueue_t>();

    auto input_boxing_data = std::make_shared<BoxingData>();
    auto output_boxing_data = std::make_shared<BoxingData>();
    auto inputMluPtrS = MakeArray<void*>(inputNum);
    auto outputMluPtrS = MakeArray<void*>(outputNum);

    cnrtGetInputDataSize(&(this->inputSizeS), &inputNum, this->function());
    for (int i = 0; i < inputNum; i++) {
      cnrtMalloc(&(inputMluPtrS.get()[i]), this->inputSizeS[i]);
    }

    cnrtGetOutputDataSize(&(this->outputSizeS), &outputNum, this->function());
    for (int i = 0; i < outputNum; i++) {
      cnrtMalloc(&(outputMluPtrS.get()[i]), this->outputSizeS[i]);
    }

    // timestamp (smart pointer) for recording starting time & ending time
    auto timestamp = std::make_shared<InferenceTimeTrace>();
    input_boxing_data->setQueue(queue);
    input_boxing_data->setBuf(inputMluPtrS);
    input_boxing_data->setStamp(timestamp);
    output_boxing_data->setBuf(outputMluPtrS);
    this->pushFreeInputBoxingData(input_boxing_data, deviceId);
    this->pushFreeOutputBoxingData(output_boxing_data, deviceId);
  }

}

#endif  // defined(USE_MLU) && defined(USE_OPENCV)
