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

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_COMMON_FUNCTIONS_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_COMMON_FUNCTIONS_HPP_
#include <glog/logging.h>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <chrono>  // NOLINT
#include <iostream>
#include <thread>
#include <command_option.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

typedef typename std::chrono::high_resolution_clock::time_point TimePoint;
typedef typename std::chrono::microseconds TimeDuration_us;
typedef typename std::chrono::milliseconds TimeDuration_ms;
typedef typename std::chrono::seconds TimeDuration_s;

void setupConfig(int threadID, int deviceID, int deviceSize);

void setDeviceId(int deviceID);

std::vector<int> getTop5(std::vector<std::string> labels,
                         std::string image,
                         float* data,
                         int count);

void readYUV(std::string name, cv::Mat img, int h, int w);

cv::Mat readImage(std::string name, cv::Size size, bool yuvImg);

cv::Mat yuv420sp2Bgr24(cv::Mat yuv_image);

cv::Mat convertYuv2Mat(std::string img_name, cv::Size inGeometry);

cv::Mat convertYuv2Mat(std::string img_name, int widht, int height);

cv::Mat scaleResizeCrop(cv::Mat sample, cv::Size inGeometry);

void printfMluTime(std::string message, float mluTime);

void printfMluTime(float mluTime);

void printfAccuracy(int imageNum, float acc1, float acc5);

void printPerf(int imageNum, float execTime, float mluTime, int batch_size = 1, int threads = 1);

void saveResult(int imageNum, float top1, float top5, float meanAp,
    float hardwaretime, float endToEndTime, int batch_size  = 1, int threads = 1);

std::string float2string(float value);

std::string double2string(const double &value);

void dumpJson(int imageNum, float top1, float top5, float meanAp,
              float latency, float throughput, float time);

class Timer {
  public:
  Timer() : time_interval_(0.) {
    start_ = std::chrono::high_resolution_clock::now();
  }
  void log(const char* msg) {
    if (time_interval_ == 0.) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> diff = end - start_;
      time_interval_ = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      LOG_IF(INFO, FLAGS_timestamp == 1)
        << "[THREAD_ID_" << std::this_thread::get_id() << "]" << "[TIMESTAMP_START_"
        << start_.time_since_epoch().count() << "]" << "[" << msg << "]\n"
        << "[THREAD_ID_" << std::this_thread::get_id() << "]" << "[TIMESTAMP_END_"
        << end.time_since_epoch().count() << "]" << "[" << msg << "]\n"
        << "[THREAD_ID_" << std::this_thread::get_id() << "]" << "[TIMESTAMP_DURATION_"
        << std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count() << "]"
        << "[" << msg << "]";
    }

    LOG(INFO) << msg << ": " << time_interval_ << " us";
  }


  float getDuration() {
    if (time_interval_ == 0.) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> diff = end - start_;
      time_interval_ = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    }
    return time_interval_;
  }

  protected:
  float time_interval_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class ImageReader {
  public:
  ImageReader(const std::string& img_path, const std::string& file_list_path,
              int thread_num = 1) : image_num_(0) {
    image_list_.resize(thread_num);
    std::ifstream file_list(file_list_path, std::ios::in);
    std::string line_tmp;
    CHECK(!file_list.fail()) << "Image file is invalid!";
    while (getline(file_list, line_tmp)) {
      line_tmp = img_path + '/' + line_tmp;
      image_list_[image_num_ % thread_num].push(line_tmp);
      image_num_++;
    }
    file_list.close();
    LOG(INFO) << "there are " << image_num_ << " figures in "
              << file_list_path;
  }
  inline std::vector<std::queue<std::string>> getImageList() { return image_list_; }
  inline int getImageNum() { return image_num_; }
  private:
    std::vector<std::queue<std::string>> image_list_;
    int image_num_;
};

struct InferenceTimeTrace {
  TimePoint in_start;
  TimePoint in_end;
  TimePoint compute_start;
  TimePoint compute_end;
  TimePoint out_start;
  TimePoint out_end;
};

class BoxingData;

TimeDuration_us getTimeDurationUs(const InferenceTimeTrace timetrace);

void printPerfTimeTraces(std::vector<InferenceTimeTrace> timetraces, int batchsize, float time = 0);

void saveResultTimeTrace(std::vector<InferenceTimeTrace> timetraces, float top1, float top5,
                         float meanAp, int imageNum, int batchsize, float time = 0);

#endif  //  CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_COMMON_FUNCTIONS_HPP_
