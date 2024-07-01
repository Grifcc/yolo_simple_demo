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

#include <sys/time.h>
#include <algorithm>
#include <atomic>
#include <condition_variable> // NOLINT
#include <thread> // NOLINT
#include <utility>
#include <string>
#include <vector>

#include "include/blocking_queue.hpp"
#include "include/queue.hpp"
#include "include/pipeline.hpp"

int Pipeline::imageNum = 0;
std::vector<std::queue<std::string>> Pipeline::imageList;
std::condition_variable Pipeline::condition;
std::mutex Pipeline::condition_m;
int Pipeline::start = 0;
std::vector<std::thread*> Pipeline::stageThreads;
std::vector<Pipeline*> Pipeline::pipelines;

Pipeline::Pipeline(DataProvider *provider,
                   Runner *runner,
                   PostProcessor *postprocessor)
                   : data_provider_(nullptr),
                     runner_(nullptr),
                     postProcessor_(nullptr) {
  data_provider_ = provider;
  runner_ = runner;
  postProcessor_ = postprocessor;

  data_provider_->setThreadId(runner_->threadId());
  postProcessor_->setThreadId(runner_->threadId());

}

Pipeline::Pipeline(const std::vector<DataProvider*>& providers,
                                 Runner *runner,
                                 PostProcessor *postprocessor)
                                 : data_provider_(nullptr),
                                   runner_(nullptr),
                                   postProcessor_(nullptr) {
  CHECK(providers.size() > 0) << "[Error]the size of providers should greater than 0.";
  runner_ = runner;
  postProcessor_ = postprocessor;

  postProcessor_->setThreadId(runner_->threadId());

  data_providers_ = providers;
  for (auto data_provider : data_providers_) {
    data_provider->setThreadId(runner_->threadId());
  }
}

Pipeline::Pipeline(const std::vector<DataProvider*>& providers,
                                 Runner *runner,
                                 const std::vector<PostProcessor*>& postprocessors)
                                 : data_provider_(nullptr),
                                   runner_(nullptr),
                                   postProcessor_(nullptr) {
  CHECK(providers.size() > 0) << "[Error]the size of providers should greater than 0.";
  runner_ = runner;
  postProcessors_ = postprocessors;

  data_providers_ = providers;
  for (auto data_provider : data_providers_) {
    data_provider->setThreadId(runner_->threadId());
    data_provider->setDeviceId(runner_->deviceId());
    data_provider->setOfflineDescripter(runner_->offlineDescripter());
  }
  for (auto postProcessor : postProcessors_) {
    postProcessor->setThreadId(runner_->threadId());
    postProcessor->setDeviceId(runner_->deviceId());
    postProcessor->setOfflineDescripter(runner_->offlineDescripter());
  }
}

Pipeline::~Pipeline() {
  // delete data_providers_ only for simple compile
  for (auto data_provider : data_providers_) {
    delete data_provider;
  }

  for (auto postProcessor : postProcessors_) {
    delete postProcessor;
  }

  // delete data_provider_ only for flexible compile
  if (data_provider_) {
    delete data_provider_;
  }

  if (runner_) {
    delete runner_;
  }

  if (postProcessor_) {
    delete postProcessor_;
  }
}

void Pipeline::runParallel() {
  int data_provider_num = (data_providers_.size() == 0) ? 1 : data_providers_.size();
  int postprocessor_num = (postProcessors_.size() == 0) ? 1 : postProcessors_.size();
  std::vector<std::thread*> threads(data_provider_num + postprocessor_num + 1, nullptr);

  for (int i = 0; i < data_provider_num; i++) {
    threads[i] = new std::thread(&DataProvider::runParallel,
                                 data_providers_[i]);
  }

  threads[data_provider_num] = new std::thread(&Runner::runParallel, runner_);

  for (int i = 0; i < postprocessor_num; i++) {
    threads[data_provider_num + 1 + i] = new std::thread(&PostProcessor::runParallel,
                                                         postProcessors_[i]);
  }

  for (int i = 0; i < data_provider_num; i++) {
    threads[i]->join();
    delete threads[i];
  }
  // push a nullptr for simple compile when the thread of data provider finished tasks
  runner_->offlineDescripter()->pushValidInputBoxingData(nullptr, runner_->deviceId());

  threads[data_provider_num]->join();
  delete threads[data_provider_num];

  for (int i = 0; i < postprocessor_num; i++) {
    runner_->offlineDescripter()->pushValidOutputBoxingData(nullptr, runner_->deviceId());
  }

  for (int i = 0; i < postprocessor_num; i++) {
    threads[data_provider_num + 1 + i]->join();
    delete threads[data_provider_num + 1 + i];
  }
}

void Pipeline::notifyAll() {
  {
    std::lock_guard<std::mutex> lk(condition_m);
    start = 1;
    LOG(INFO) << "Notify to start ...";
  }
  condition.notify_all();
}

void Pipeline::waitForNotification() {
  std::unique_lock<std::mutex> lk(condition_m);
  LOG(INFO) << "Waiting ...";
  condition.wait(lk, [](){return start;});
  lk.unlock();
}
