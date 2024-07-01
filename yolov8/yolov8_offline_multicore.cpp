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
// #define USE_MLU
// #define USE_OPENCV
#if defined(USE_MLU) && defined(USE_OPENCV)
#include <sys/time.h>
#include <gflags/gflags.h>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include "pipeline.hpp"
#include "off_data_provider.hpp"
#include "off_runner.hpp"
#include "yolov8_off_post.hpp"
#include "common_functions.hpp"
#include "simple_interface.hpp"

DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");
DEFINE_int32(first_conv, 1, "0 or 1, use first conv or not.");
DEFINE_string(outputdir, ".", "The directoy used to save output images");

template <class Dtype, template <class...> class Qtype>
class OffYolov8DataProvider : public OffDataProvider<Dtype, Qtype> {
private:
    int first_conv_;
public:
    explicit OffYolov8DataProvider(const std::string& meanvalue,
                                   const std::string& stdvalue,
                            const std::queue<std::string>& images,
                            const int first_conv = 0)
        : OffDataProvider<Dtype, Qtype>(meanvalue, stdvalue, images), first_conv_(first_conv) {}
    void resizeMat(const cv::Mat& sample, cv::Mat& sample_resized) {
      int input_dim = this->runner_->h();
      if (sample.size() == this->inGeometry_) {
        sample_resized = sample;
      } else {
        float img_w = sample.cols;
        float img_h = sample.rows;
        int tmp_dim = img_w > img_h ? img_w : img_h;
        cv::Mat tmp_resized(tmp_dim, tmp_dim, CV_8UC3,
                            cv::Scalar(114, 114, 114));
        sample.copyTo(tmp_resized(
                      cv::Range((static_cast<float>(tmp_dim) - img_h) / 2,
                                (static_cast<float>(tmp_dim) - img_h) / 2 + img_h),
                      cv::Range((static_cast<float>(tmp_dim) - img_w) / 2,
                                (static_cast<float>(tmp_dim) - img_w) / 2 + img_w)));

        cv::resize(tmp_resized, sample_resized, cv::Size(input_dim, input_dim), cv::INTER_AREA);
      }
    }

    void normalizeMat(const cv::Mat& sample_float, cv::Mat& sample_normalized) {
      if(first_conv_ == 1)
      {
        sample_normalized = sample_float;
      }else
      {
        sample_normalized = sample_float / 255.;
      }
    }
};  //  OffYolov8DataProvider

typedef DataProvider<void*, BlockingQueue> DataProviderT;
typedef OffYolov8DataProvider<void*, BlockingQueue> OffYolov8DataProviderT;
typedef OffRunner<void*, BlockingQueue> OffRunnerT;
typedef PostProcessor<void*, BlockingQueue> PostProcessorT;
typedef YoloV8OffPostProcessor<void*, BlockingQueue> YoloV8OffPostProcessorT;
typedef Pipeline<void*, BlockingQueue> PipelineT;

int main(int argc, char* argv[]) {
  {
    const char * env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0)
      FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Do detection using yolov5 mode.\n"
        "Usage:\n"
        "    yolov8_offline_multicore [FLAGS] model_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(argv[0],
                "examples/offline/yolov8/yolov8_offline_multicore");
    return 1;
  }

  auto& simpleInterface = SimpleInterface::getInstance();
  int provider_num = 1, postprocessor_num = 1;
  simpleInterface.setFlag(true);
  provider_num = SimpleInterface::data_provider_num_;
  postprocessor_num = SimpleInterface::postProcessor_num_;

  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  // get device ids
  std::stringstream sdevice(FLAGS_mludevice);
  std::vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }

  int totalThreads = FLAGS_threads * deviceIds_.size();

  cnrtInit(0);
  simpleInterface.loadOfflinemodel(FLAGS_offlinemodel,
                                   deviceIds_,
                                   FLAGS_channel_dup,
                                   FLAGS_threads);

  ImageReader img_reader(FLAGS_dataset_path, FLAGS_images, totalThreads * provider_num);
  auto&& imageList = img_reader.getImageList();
  int imageNum = img_reader.getImageNum();
  if (FLAGS_perf_mode) {
    // calculate number of fake image per thread
    FLAGS_perf_mode_img_num = FLAGS_perf_mode_img_num / (totalThreads * provider_num);
  }
  std::vector<std::thread*> stageThreads;
  std::vector<PipelineT*> pipelines;
  std::vector<DataProviderT*> providers;
  std::vector<PostProcessorT*> postprocessors;
  for (int i = 0; i < totalThreads; i++) {
    DataProviderT* provider;
    OffRunnerT* runner;
    PipelineT* pipeline;
    PostProcessorT* postprocessor;

    providers.clear();
    postprocessors.clear();
    // provider_num is 1 for flexible compile.
    for (int j = 0; j < provider_num; j++) {
      provider = new OffYolov8DataProviderT(FLAGS_meanvalue, FLAGS_stdvalue,
                                      imageList[provider_num * i + j],
                                      FLAGS_first_conv);
      providers.push_back(provider);
    }


    for (int j = 0; j < postprocessor_num; j++) {
      postprocessor = new YoloV8OffPostProcessorT();
      postprocessors.push_back(postprocessor);
    }

    auto dev_runtime_contexts = simpleInterface.get_runtime_contexts();
    int index = i % deviceIds_.size();
    int thread_id = i / deviceIds_.size();
    runner = new OffRunnerT(dev_runtime_contexts[index][thread_id], i);
    pipeline = new PipelineT(providers, runner, postprocessors);

    stageThreads.push_back(new std::thread(&PipelineT::runParallel, pipeline));
    pipelines.push_back(pipeline);
  }

  Timer timer;
  for (int i = 0; i < stageThreads.size(); i++) {
    pipelines[i]->notifyAll();
  }

  for (int i = 0; i < stageThreads.size(); i++) {
    stageThreads[i]->join();
    delete stageThreads[i];
  }
  timer.log("Total execution time");

  float mluTime = 0;
  for (auto pipeline : pipelines) {
    mluTime += pipeline->runner()->runTime();
  }
  int batch_size = pipelines[0]->runner()->n();
  std::vector<InferenceTimeTrace> timetraces;
  for (auto iter : pipelines) {
    for (auto pP : iter->postProcessors()) {
      for (auto tc : pP->timeTraces()) {
        timetraces.push_back(tc);
      }
    }
  }
  printPerfTimeTraces(timetraces, batch_size, mluTime);
  saveResultTimeTrace(timetraces, (-1), (-1), (-1), imageNum, batch_size, mluTime);

  for (auto pipeline : pipelines)
    delete pipeline;
  simpleInterface.destroyRuntimeContext();
  cnrtDestroy();
}

#else
#include <glog/logging.h>
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             <<" of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
