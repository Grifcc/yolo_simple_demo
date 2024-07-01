/*
All modification made by Cambricon Corporation: © 2019 Cambricon Corporation
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

#if defined(USE_MLU) && defined(USE_OPENCV)
#include <queue>
#include <string>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <iomanip>
#include "cnrt.h" // NOLINT
#include "glog/logging.h"
#include "yolov5_off_post.hpp"
#include "off_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"
#include "threadPool.h"

template<typename Dtype, template <class...> class Qtype>
std::mutex PostProcessor<Dtype, Qtype>::post_mutex_;

template<typename Dtype, template <typename> class Qtype>
void YoloV5OffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> * infr =
      static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  setDeviceId(infr->deviceId());

  this->readLabels(&this->label_to_display_name);
  int outputNum = infr->outNum();
  std::vector<std::vector<int>> out_shapes;
  for (int i = 0; i < outputNum; i++) {
    unsigned int n, c, h, w;
    auto shape = std::make_shared<int *>();
    int dimNum = 4;
    cnrtGetOutputDataShape(shape.get(), &dimNum, 0, infr->function());  // NCHW
    n = shape.get()[0][0];
    h = shape.get()[0][1];
    w = shape.get()[0][2];
    c = shape.get()[0][3];
    free(shape.get()[0]);  // cnrtGetOutputDataShape malloc for shape which need free outside.
    std::vector<int> out_shape;
    out_shape.push_back(n);
    out_shape.push_back(c);
    out_shape.push_back(h);
    out_shape.push_back(w);
    out_shapes.push_back(out_shape);
  }
  int outShape = out_shapes[0][0];

  cnrtDataType_t* output_data_type = NULL;
  cnrtGetOutputDataType(&output_data_type, &outputNum, infr->function());
  int dim_order[4] = {0, 3, 1, 2};

  outCpuPtrs_ = new(Dtype);
  outCpuPtrs_[0] = new float[infr->outCount()];
  outTrans_ = new(Dtype);
  outTrans_[0] = new float[infr->outCount()];
  outTempCpuPtrs_ = new(Dtype);
  outTempCpuPtrs_[0] =
    new char[GET_OUT_TENSOR_SIZE(output_data_type[0], infr->outCount())];

  int TASK_NUM = SimpleInterface::thread_num;
  zl::ThreadPool tp(SimpleInterface::thread_num);
  while (true) {
    Timer postProcess;
    std::unique_lock<std::mutex> lock(PostProcessor<Dtype, Qtype>::post_mutex_);
    std::shared_ptr<Dtype> mluOutData = infr->popValidOutputData();
    if (mluOutData == nullptr) {
      lock.unlock();
      break;  // no more work
    }
    auto&& origin_img = infr->popValidInputNames();
    lock.unlock();

    Dtype* cpuOutTempData = outTempCpuPtrs_;
    Timer copyout;
    TimePoint t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < outputNum; i++) {
      cnrtMemcpy(cpuOutTempData[i],
                 mluOutData.get()[i],
                 infr->outputSizeS[i],
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
      if (output_data_type[i] != CNRT_FLOAT32) {
        cnrtCastDataType(cpuOutTempData[i],
                         output_data_type[i],
                         outTrans_[i],
                         CNRT_FLOAT32,
                         infr->outCount(),
                         NULL);
      } else {
        memcpy(outTrans_[i], cpuOutTempData[i], infr->outputSizeS[i]);
      }

      int dim_shape[4] = {out_shapes[i][0], out_shapes[i][2],
                          out_shapes[i][3], out_shapes[i][1]};
      cnrtTransDataOrder(outTrans_[i],
                         CNRT_FLOAT32,
                         outCpuPtrs_[i],
                         4,
                         (int*)dim_shape,
                         dim_order);
    }

    copyout.log("copyout time ...");
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = infr->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    infr->pushFreeInputTimeTraceData(timetrace);
    infr->pushFreeOutputData(mluOutData);

    std::vector<cv::Mat> imgs;
    std::vector<std::string> img_names;
    auto boxes = getResults(outShape);
    Timer dumpTimer;
    if (FLAGS_dump && !FLAGS_perf_mode) {
      getImages(&imgs, &img_names, origin_img);
      const int size = imgs.size();
      if (TASK_NUM > size)
        TASK_NUM = size;
      const int delta = size / TASK_NUM;
      int from = 0;
      int to = delta;
      for (int i = 0; i < TASK_NUM; i++) {
        from = delta * i;
        if (i == TASK_NUM - 1) {
          to = size;
        } else {
          to = delta * (i + 1);
        }

        tp.add([](const std::vector<cv::Mat>& imgs,
                  const std::vector<std::vector<std::vector<float>>>& boxes,
                  const std::vector<std::string>& label_to_display_name,
                  const std::vector<std::string>& img_names,
                  const int& from, const int& to, YoloV5Processor<Dtype, Qtype>* object) {
                    object->WriteVisualizeBBox_offline(imgs, boxes,
                                label_to_display_name, img_names,
                                from, to);}, imgs, boxes,
                  this->label_to_display_name, img_names, from, to, this);
      }
    }
    dumpTimer.log("dump out time ...");
    postProcess.log("post process time ...");
  }
}

template<typename Dtype, template <typename> class Qtype>
std::vector<std::vector<std::vector<float>>> YoloV5OffPostProcessor<Dtype, Qtype>::getResults(int outShape) {
  OffRunner<Dtype, Qtype> * infr =
      static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  int outN = infr->outNum();
  int outCount = infr->outCount();
  int sBatchsize = outCount / outN;

  float* data = reinterpret_cast<float*>(outCpuPtrs_[0]);
  std::vector<std::vector<std::vector<float>>> boxes = this->detection_out(data, outShape,
                                                     sBatchsize, outN);
  return boxes;
}

template<typename Dtype, template<typename> class Qtype>
void YoloV5OffPostProcessor<Dtype, Qtype>::getImages(
    std::vector<cv::Mat> *imgs,
    std::vector<std::string> *img_names,
    const std::vector<std::pair<std::string, cv::Mat>> &origin_img) {

  OffRunner<Dtype, Qtype> * infr = static_cast<OffRunner<Dtype, Qtype>*>(this->runner_);
  for (auto& img_name : origin_img) {
    if (img_name.first != "null") {
      cv::Mat img;
      if (FLAGS_yuv) {
        img = convertYuv2Mat(img_name.first, infr->w(), infr->h());
      } else {
        /* img = cv::imread(img_name, -1); */
        img = img_name.second;
      }
      int pos = img_name.first.find_last_of('/');
      std::string file_name(img_name.first.substr(pos+1));
      imgs->push_back(img);
      img_names->push_back(file_name);
    }
  }
}

INSTANTIATE_OFF_CLASS(YoloV5OffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
