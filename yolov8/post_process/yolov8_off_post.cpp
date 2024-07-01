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
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "cnrt.h" // NOLINT
#include "glog/logging.h"
#include "yolov8_off_post.hpp"
#include "off_runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"
#include "threadPool.h"


// #define NMS

template<typename Dtype, template <class...> class Qtype>
std::mutex PostProcessor<Dtype, Qtype>::post_mutex_;

template<typename Dtype, template <typename> class Qtype>
void YoloV8OffPostProcessor<Dtype, Qtype>::runParallel() {
  OffRunner<Dtype, Qtype> *infr =
      static_cast<OffRunner<Dtype, Qtype> *>(this->runner_);
  setDeviceId(infr->deviceId());

  this->readLabels(&this->label_to_display_name);
  // int outputNum = infr->outNum();
  int outputNum = infr->outBlobNum();
  std::vector<std::vector<int>> out_shapes;
  for (int i = 0; i < outputNum; i++) {
    unsigned int n, c, h, w;
    auto shape = std::make_shared<int *>();
    int dimNum = 4;
    cnrtGetOutputDataShape(shape.get(), &dimNum, i, infr->function()); // NCHW
    n = shape.get()[0][0];
    h = shape.get()[0][1];
    w = shape.get()[0][2];
    c = shape.get()[0][3];
    free(shape.get()[0]); // cnrtGetOutputDataShape malloc for shape which need free outside.
    std::vector<int> out_shape;
    out_shape.push_back(n);
    out_shape.push_back(c);
    out_shape.push_back(h);
    out_shape.push_back(w);
    out_shapes.push_back(out_shape);
  }


  cnrtDataType_t *output_data_type = NULL;
  cnrtGetOutputDataType(&output_data_type, &outputNum, infr->function());
  int dim_order[4] = {0, 3, 1, 2};


  outCpuPtrs_.resize(outputNum);
  outTrans_.resize(outputNum);
  outTempCpuPtrs_.resize(outputNum);

  for (size_t i = 0; i < outputNum; i++)
  {
    outCpuPtrs_[i] = new float[infr->outAllCounts()[i]];
    outTrans_[i] = new float[infr->outAllCounts()[i]];
    size_t __size = GET_OUT_TENSOR_SIZE(output_data_type[i], infr->outAllCounts()[i]);
    outTempCpuPtrs_[i] =
        new char[__size];
  }

  int TASK_NUM = SimpleInterface::thread_num;
  zl::ThreadPool tp(SimpleInterface::thread_num);
  while (true) {
    Timer postProcess;


    std::unique_lock<std::mutex> lock(PostProcessor<Dtype, Qtype>::post_mutex_);
    Timer start;
    std::shared_ptr<Dtype> mluOutData = infr->popValidOutputData();
    start.log("start time ...");
    if (mluOutData == nullptr) {
      lock.unlock();
      break;  // no more work
    }

    auto&& origin_img = infr->popValidInputNames();
    lock.unlock();
    // Dtype* outTempCpuPtrs_ = outTempCpuPtrs_;
    Timer copyout;
    TimePoint t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < outputNum; i++) {
      cnrtMemcpy(outTempCpuPtrs_[i],
                 mluOutData.get()[i],
                 infr->outputSizeS[i],
                 CNRT_MEM_TRANS_DIR_DEV2HOST);
      if (output_data_type[i] != CNRT_FLOAT32) {
        long size = infr->outAllCounts()[i];
        cnrtCastDataType(outTempCpuPtrs_[i],
                         output_data_type[i],
                         outTrans_[i],
                         CNRT_FLOAT32,
                         size,
                         NULL);
      } else {
        memcpy(outTrans_[i], outTempCpuPtrs_[i], infr->outputSizeS[i]);
      }

      int dim_shape[4] = {out_shapes[i][0], out_shapes[i][2],
                          out_shapes[i][3], out_shapes[i][1]};
      cnrtTransDataOrder(outTrans_[i],
                         CNRT_FLOAT32,
                         outCpuPtrs_[i],
                         4,
                         (int *)dim_shape,
                         dim_order);
    }

    copyout.log("copyout time ...");
    infr->pushFreeOutputData(mluOutData);

    std::vector<cv::Mat> imgs; 
    std::vector<std::string> img_names;

    std::vector<std::vector<std::vector<float>>> dbox;
    std::vector<std::vector<std::vector<float>>> boxes;
    std::vector<std::vector<std::vector<float>>> nms_boxes;

    Timer processout;

    make_anchors(anchors, strides, out_shapes[0][1], out_shapes[0][2], 0.5);
    dbox = dist2bbox(outCpuPtrs_[0], anchors, strides, out_shapes[0][1], out_shapes[0][2], out_shapes[0][0]);
    boxes = process(outCpuPtrs_[1], dbox, out_shapes[1][1], out_shapes[0][1], out_shapes[0][2], out_shapes[0][0],this->conf_thres);
    nms_boxes = nonMaximumSuppression(boxes, this->iou_thres, this->nc);

    processout.log("processout time ...");


    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto timetrace = infr->popValidOutputTimeTraceData();
    timetrace->out_start = t1;
    timetrace->out_end = t2;
    this->appendTimeTrace(*timetrace);
    infr->pushFreeInputTimeTraceData(timetrace);


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
                  const std::vector<std::vector<std::vector<float>>>& nms_boxes,
                  const std::vector<std::string>& label_to_display_name,
                  const std::vector<std::string>& img_names,
                  const int& from, const int& to, YoloV8Processor<Dtype, Qtype>* object) {
                    object->WriteVisualizeBBox_offline(imgs, nms_boxes,
                                label_to_display_name, img_names,
                                from, to);}, imgs, nms_boxes,
                  this->label_to_display_name, img_names, from, to, this);
      }
    }
    dumpTimer.log("dump out time ...");
    postProcess.log("post process time ...");
  }
}

template<typename Dtype, template <typename> class Qtype>
std::vector<std::vector<std::vector<float>>> YoloV8OffPostProcessor<Dtype, Qtype>::getResults(int outShape) {
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
void YoloV8OffPostProcessor<Dtype, Qtype>::getImages(
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


template <typename Dtype, template <typename> class Qtype>
void YoloV8OffPostProcessor<Dtype, Qtype>::make_anchors(std::vector<std::vector<float>> &anchors,
                                                     std::vector<float> &strides,
                                                     const int grid_h,
                                                     const int grid_w,
                                                     const float grid_cell_offset)
{
  anchors.resize(2);
  for (int i = 0; i < 6400; i++) {
    anchors[0].emplace_back(i%80 + grid_cell_offset) ;  
    anchors[1].emplace_back(i/80 + grid_cell_offset) ;  
  }
  for (int i = 0; i < 1600; i++) {
    anchors[0].emplace_back(i%40 + grid_cell_offset) ;  
    anchors[1].emplace_back(i/40 + grid_cell_offset) ;  
  }
  for (int i = 0; i < 400; i++) {
    anchors[0].emplace_back(i%20 + grid_cell_offset) ;
    anchors[1].emplace_back(i/20 + grid_cell_offset) ;
  }

  
  for (int i = 0; i < 6400; i++){
    strides.emplace_back(8.);
  }
  for (int i = 0; i < 1600; i++)
  {
    strides.emplace_back(16.);
  }
  for (int i = 0; i < 400; i++)
  {
    strides.emplace_back(32.);
  }
}


template <typename Dtype, template <typename> class Qtype>
std::vector<std::vector<std::vector<float>>> YoloV8OffPostProcessor<Dtype, Qtype>::dist2bbox(float *input,
                                                     const std::vector<std::vector<float>> anchors,
                                                     const std::vector<float> strides,
                                                     const int grid_h,
                                                     const int grid_w,
                                                     const int batch
                                                     )
{

  std::vector<std::vector<std::vector<float>>> box(batch, std::vector<std::vector<float>>(grid_h, std::vector<float>(grid_w)));
  std::vector<std::vector<std::vector<float>>> dbox(batch, std::vector<std::vector<float>>(grid_h, std::vector<float>(grid_w)));

  
  int offset = 0;
  for (int n = 0; n < batch; n++)
  {
    for (int i = 0; i < grid_h; i++)
    {
      for (int j = 0; j < grid_w; j++)
      {
        float *val = input + offset;
        box[n][i][j] = *val;
        offset += 1;

        if(i < grid_h/2){
          box[n][i][j] = anchors[i%2][j] - box[n][i][j];//x1y1
        }
        else {
          box[n][i][j] = anchors[i%2][j] + box[n][i][j];//x2y2
        }

      }
    }
  }

  for (int n = 0; n < batch; n++)
  {
    for (int i = 0; i < grid_h; i++)
    {
      for (int j = 0; j < grid_w; j++)
      {
        if(i < grid_h/2){
          dbox[n][i][j] = ((box[n][i][j] + box[n][i+grid_h/2][j]) / 2) * strides[j];//x1y1+x2y2
        }
        else {
          dbox[n][i][j] = (box[n][i][j] - box[n][i-grid_h/2][j]) * strides[j];//x2y2 -x1y1
        }
      }
    }
  }
  
  return dbox;
}

template <typename Dtype, template <typename> class Qtype>
std::vector<std::vector<std::vector<float>>> YoloV8OffPostProcessor<Dtype, Qtype>::process(float *input,
                                                    std::vector<std::vector<std::vector<float>>> dbox, 
                                                    const int cls_grid_h,
                                                    const int box_grid_h,
                                                    const int grid_w,
                                                    const int batch,
                                                    const float conf_thres
                                                    )
{
  std::vector<std::vector<std::vector<float>>> allboxes(batch);
  std::vector<std::vector<std::vector<float>>> cls(batch, std::vector<std::vector<float>>(cls_grid_h, std::vector<float>(grid_w)));

  int offset = 0;
  for (int n = 0; n < batch; n++)
  {
    for (int i = 0; i < cls_grid_h; i++)
    {
      for (int j = 0; j < grid_w; j++)
      {
        float *val = input + offset;
        cls[n][i][j] = *val;
        offset += 1;
      } 
    }
  }
  
  for (int n = 0; n < batch; n++)
  {
    for (int j = 0; j < grid_w; j++)
    {
      float max = 0.0;
      float max_id = 0.0;
      for (int i = 0; i < cls_grid_h; i++)
      {
        if (max < cls[n][i][j]){
          max = cls[n][i][j];
          max_id = i;
        }
      }
      if (max > conf_thres){
        allboxes[n].emplace_back(std::vector<float>());
        float x1 = dbox[n][0][j]-dbox[n][2][j]/2;
        float y1 = dbox[n][1][j]-dbox[n][3][j]/2;
        float x2 = dbox[n][0][j]+dbox[n][2][j]/2;
        float y2 = dbox[n][1][j]+dbox[n][3][j]/2;
        allboxes[n][allboxes[n].size() - 1].emplace_back(x1);
        allboxes[n][allboxes[n].size() - 1].emplace_back(y1);
        allboxes[n][allboxes[n].size() - 1].emplace_back(x2);
        allboxes[n][allboxes[n].size() - 1].emplace_back(y2);
        allboxes[n][allboxes[n].size() - 1].emplace_back(max);
        allboxes[n][allboxes[n].size() - 1].emplace_back(max_id);
      }
    }
  }
  return allboxes;
}


float iou_calculate(cv::Rect rect1, cv::Rect rect2){
  cv::Rect rect_or = rect1 | rect2; 
	cv::Rect rect_and = rect1 & rect2;
	float iou = rect_and.area() *1.0/ rect_or.area();
  return iou;
}

bool compareVec(const std::vector<float>& v1, const std::vector<float>& v2) {
    return v1[4] > v2[4];
}

template <typename Dtype, template <typename> class Qtype>
std::vector<std::vector<std::vector<float>>> YoloV8OffPostProcessor<Dtype, Qtype>::nonMaximumSuppression(std::vector<std::vector<std::vector<float>>> boxes, 
                                                                                                        const float iou_thres, 
                                                                                                        const int classnum)
{

    std::vector<std::vector<std::vector<float>>> nms_boxes;

    for (int n = 0; n < boxes.size(); n++)
    {
      std::vector<std::vector<std::vector<float>>> input_boxes(classnum);
      std::vector<std::vector<float>> output_boxes;
      for (int i = 0; i < boxes[n].size(); i++){
          if(boxes[n][i][5]<classnum){
              input_boxes[boxes[n][i][5]].push_back(boxes[n][i]);
          }
          else{
              continue;
          }
      }

      for (int i = 0; i < input_boxes.size(); i++)
      {
          std::sort(input_boxes[i].begin(), input_boxes[i].end(), compareVec);
      }
      
      for (int i = 0; i < input_boxes.size(); i++)
      {
          while (input_boxes[i].size() > 0)
          {   
              int j = 1;
              while(input_boxes[i].size() - 1 >= j)
              {
                  cv::Rect2f maxRect(input_boxes[i][0][0], input_boxes[i][0][1], input_boxes[i][0][2] - input_boxes[i][0][0], input_boxes[i][0][3] - input_boxes[i][0][1]);
                  cv::Rect2f otherRect(input_boxes[i][j][0], input_boxes[i][j][1], input_boxes[i][j][2] - input_boxes[i][j][0], input_boxes[i][j][3] - input_boxes[i][j][1]);
                  float iou = iou_calculate(maxRect, otherRect);
                  if(iou > iou_thres){
                      input_boxes[i].erase(input_boxes[i].begin()+j);
                      j--;
                  }
                  j++;
              }
              output_boxes.push_back(input_boxes[i][0]);
              input_boxes[i].erase(input_boxes[i].begin());
          }
      }
      nms_boxes.push_back(output_boxes);
    }

  
  return nms_boxes;
    
}

INSTANTIATE_OFF_CLASS(YoloV8OffPostProcessor);
#endif  // defined(USE_MLU) && defined(USE_OPENCV)
