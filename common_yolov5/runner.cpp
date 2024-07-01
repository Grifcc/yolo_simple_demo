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
#include "include/command_option.hpp"
#include "include/common_functions.hpp"
#include "include/handler.hpp"

Runner::Runner(OfflineDescripter* offline_descripter,
                                   const int& id) {
  this->threadId_ = id;
  this->runTime_ = 0;
  this->offline_descripter_ = offline_descripter;
  // Here, deviceId is tricky to keep consistance on DataProvider, PostProcessor
  // and Runner which are in same Pipeline. Runner initialize deviceId here, and
  // set id to DataProvider and PostProcessor.
  cnrtGetRuntimeContextInfo(offline_descripter->getRuntimeContext(id),
                            CNRT_RT_CTX_DEV_ORDINAL,
                            reinterpret_cast<void **>(&this->deviceId_));
}

Runner::~Runner() {
}


void Runner::runParallel() {
#ifdef PINGPONG
#define RES_SIZE 2
#else
#define RES_SIZE 1
#endif

  // set device to runtime context binded device
  cnrtRuntimeContext_t ctxs[RES_SIZE];
  ctxs[0] = this->offlineDescripter()->getRuntimeContext(threadId_);
  cnrtSetCurrentContextDevice(ctxs[0]);
  // TODO: queue with shared_ptr may need consider more. And also FLAGS_async_copy
  std::vector<std::shared_ptr<cnrtQueue_t>> queue;
  queue.push_back(std::make_shared<cnrtQueue_t>());
  for (int i = 1; i < RES_SIZE; i++) {
    queue.push_back(std::make_shared<cnrtQueue_t>());
    cnrtForkRuntimeContext(&ctxs[i], ctxs[0], NULL);
    cnrtSetCurrentContextDevice(ctxs[i]);
  }

  cnrtNotifier_t notifierBeginning[RES_SIZE];
  cnrtNotifier_t notifierEnd[RES_SIZE];

  for (int i = 0; i < RES_SIZE; i++) {
    // this queue is for sync copy, not for async copy
    if (!FLAGS_async_copy) {
      CHECK(cnrtCreateQueue(queue[i].get()) == CNRT_RET_SUCCESS)
            << "CNRT create queue error, thread_id " << this->threadId();
    }
    cnrtCreateNotifier(&notifierBeginning[i]);
    cnrtCreateNotifier(&notifierEnd[i]);
  }
  float eventInterval[RES_SIZE] = {0};
  std::shared_ptr<void*> mluInData[RES_SIZE];
  std::shared_ptr<void*> mluOutData[RES_SIZE];
  TimePoint timepoints[RES_SIZE];
  std::shared_ptr<BoxingData> input_boxing_data;
  std::shared_ptr<BoxingData> output_boxing_data;

  auto do_pop = [&](int index, void **param) {
    input_boxing_data = this->offlineDescripter()->popValidInputBoxingData(this->deviceId_);
    output_boxing_data = this->offlineDescripter()->popFreeOutputBoxingData(this->deviceId_);
    // nullptr is the last element in queue indicates all data is processed
    if ( input_boxing_data == nullptr) return false;
    mluInData[index] = input_boxing_data->getBuf();
    if (FLAGS_async_copy) {
      // TODO: Have to do if here. Else getqueue returns nullptr in normal mode
      // which cause a core dump. needs to consider a better solution
      queue[index] = input_boxing_data->getQueue();
    }

    mluOutData[index] = output_boxing_data->getBuf();
    for (int i = 0; i < offlineDescripter()->inputNum(); i++) {
      param[i] = mluInData[index].get()[i];
    }
    for (int i = 0; i < offlineDescripter()->outputNum(); i++) {
      param[offlineDescripter()->inputNum() + i] = mluOutData[index].get()[i];
    }

    return true;
  };

  auto do_invoke = [&](int index, void** param) {
    cnrtPlaceNotifier(notifierBeginning[index], *(queue[index]));
    int channel = this->threadId() % 4;
    cnrtSetCurrentChannel((cnrtChannelType_t)channel);
    CNRT_CHECK(cnrtInvokeRuntimeContext(ctxs[index],
               param, *(queue[index]), nullptr));
    timepoints[index] = std::chrono::high_resolution_clock::now();
  };

  auto do_sync = [&](int index) {
    cnrtPlaceNotifier(notifierEnd[index], *(queue[index]));
    if (cnrtSyncQueue(*(queue[index])) == CNRT_RET_SUCCESS) {
      cnrtNotifierDuration(notifierBeginning[index],
             notifierEnd[index], &eventInterval[index]);
      this->setRunTime(this->runTime() + eventInterval[index]);
      printfMluTime(eventInterval[index]);
    } else {
      LOG(ERROR) << " SyncQueue error";
    }
    this->offlineDescripter()->pushFreeInputBoxingData(input_boxing_data, this->deviceId_);
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto time_stamp = input_boxing_data->getStamp();
    time_stamp->compute_start = timepoints[index];
    time_stamp->compute_end = t2;
    output_boxing_data->setStamp(time_stamp);

    // TODO: consider more efficient copy or move
    output_boxing_data->setImageAndName(input_boxing_data->getImageAndName());
    output_boxing_data->setFeatureName(input_boxing_data->getFeatureName());
    this->offlineDescripter()->pushValidOutputBoxingData(output_boxing_data, this->deviceId_);
  };

#ifdef PINGPONG
  bool pong_valid = false;
  while (true) {
    void* param[offlineDescripter()->inputNum() + offlineDescripter()->outputNum()];

    // pop - ping
    Timer runner_timer;
    if (do_pop(0, static_cast<void **>(param)) == false) {
      if (pong_valid)
        do_sync(1);
      break;
    }

    // invoke - ping
    do_invoke(0, static_cast<void **>(param));

    // sync - pong
    if (pong_valid)
      do_sync(1);

    // pop - pong
    if (do_pop(1, static_cast<void **>(param)) == false) {
      do_sync(0);
      break;
    }

    // invoke - pong
    do_invoke(1, static_cast<void **>(param));
    pong_valid = true;

    // sync - ping
    do_sync(0);
    runner_timer.log("compute time ...");
  }
#else
  while (true) {
    void* param[offlineDescripter()->inputNum() + offlineDescripter()->outputNum()];
    // pop - ping
    Timer runner_timer;
    if (do_pop(0, static_cast<void **>(param)) == false) {
      break;
    }
    // invoke - ping
    do_invoke(0, static_cast<void **>(param));

    // sync - ping
    do_sync(0);
    runner_timer.log("compute time ...");
  }
#endif

  for (int i = 0; i < RES_SIZE; i++) {
    cnrtDestroyNotifier(&notifierBeginning[i]);
    cnrtDestroyNotifier(&notifierEnd[i]);
    // this queue is for sync copy, not for async copy
    if (!FLAGS_async_copy) {
      cnrtDestroyQueue(*(queue[i]));
    }
  }
  // tell postprocessor to exit
}

#endif  // defined(USE_MLU) && defined(USE_OPENCV)
