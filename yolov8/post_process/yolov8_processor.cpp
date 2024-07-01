

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

#include "glog/logging.h"
#ifdef USE_MLU
#include <algorithm>
#include "cnrt.h" // NOLINT
#include "runner.hpp"
#include "yolov8_processor.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

float clip(float x,float min,float max)
{
   return x>min? (x<max?x:max):min;
}
template <typename Dtype, template <typename> class Qtype>
std::vector<std::vector<std::vector<float>>> YoloV8Processor<Dtype, Qtype>::detection_out(
    float* outputData, int batchSize, int sBatchsize, int outN) {

    std::vector<std::vector<std::vector<float>>> final_boxes(outN);
    int numBoxFinal = 0;
    int imgSize = 640;
    float max_limit = imgSize;
    float min_limit = 0;

    for (int i = 0; i < batchSize; i++) {
      numBoxFinal = (int)outputData[i*sBatchsize];
      for (int k = 0; k < numBoxFinal ;  k++) {
        std::vector<float> single_box;
        int batchNum = outputData[i * sBatchsize + 64 + k * 7];
        if ((batchNum < 0) || (batchNum >= batchSize)) {
          continue;
        }
        float bl = std::max(min_limit,
                            std::min(max_limit,
                                     outputData[i * sBatchsize + 64 + k * 7 + 3]));
        float br = std::max(min_limit,
                            std::min(max_limit,
                                     outputData[i * sBatchsize + 64 + k * 7 + 4]));
        float bt = std::max(min_limit,
                            std::min(max_limit,
                                     outputData[i * sBatchsize + 64 + k * 7 + 5]));
        float bb = std::max(min_limit,
                            std::min(max_limit,
                                     outputData[i * sBatchsize + 64 + k * 7 + 6]));
        single_box.push_back(bl);
        single_box.push_back(br);
        single_box.push_back(bt);
        single_box.push_back(bb);
        single_box.push_back(outputData[i * sBatchsize + 64 + k * 7 + 2]);
        single_box.push_back(outputData[i * sBatchsize + 64 + k * 7 + 1]);
        if ((bt - bl) > 0 && (bb - br) > 0) {
          final_boxes[batchNum].push_back(single_box);
        }
      }
    }
    return final_boxes;
}

template <typename Dtype, template <typename> class Qtype>
void YoloV8Processor<Dtype, Qtype>::readLabels(std::vector<std::string>* labels) {
  if (!FLAGS_labels.empty()) {
    std::ifstream file(FLAGS_labels);
    std::string line;
    while (std::getline(file, line)) {
      labels->push_back(line);
    }
    file.close();
  }
}

template <typename Dtype, template <typename> class Qtype>
void YoloV8Processor<Dtype, Qtype>::WriteVisualizeBBox_offline(
    const std::vector<cv::Mat>& images,
    const std::vector<std::vector<std::vector<float>>>& detections,
    const std::vector<std::string>& labelToDisplayName,
    const std::vector<std::string>& imageNames,
    const int from, const int to) {
    // Retrieve detections.
    for (int i = from; i < to; ++i) {
      if (imageNames[i] == "null") continue;
      cv::Mat image;
      image = images[i];
      std::vector<std::vector<float>> result = detections[i];
      std::string name = imageNames[i];
      int positionMap = imageNames[i].rfind("/");
      if (positionMap > 0 && positionMap < imageNames[i].size()) {
        name = name.substr(positionMap + 1);
      }
      positionMap = name.find(".");
      if (positionMap > 0 && positionMap < name.size()) {
        name = name.substr(0, positionMap);
      }
      std::string filename = name + ".txt";
      std::ofstream fileMap(FLAGS_outputdir + "/" + filename);
      float img_max_shape = (image.cols < image.rows)?image.rows:image.cols;
      float pad_x = (image.rows > image.cols)?(image.rows - image.cols):0;
      float pad_y = (image.rows < image.cols)?(image.cols - image.rows):1;
      float img_size = 640;
      pad_x = pad_x * (img_size/img_max_shape);
      pad_y = pad_y * (img_size/img_max_shape);
      float unpad_h = img_size - pad_y;
      float unpad_w = img_size - pad_x;
      for (int j = 0; j < result.size(); j++) {
        float box_h = ((result[j][3] - result[j][1])/unpad_h) * image.rows;
        float box_w = ((result[j][2] - result[j][0])/unpad_w) * image.cols;
        float x0 =((result[j][0] - std::floor(pad_x/2))/unpad_w) * image.cols;
        float y0 =((result[j][1] - std::floor(pad_y/2))/unpad_h) * image.rows;

        float x1 = x0 + box_w;
        float y1 = y0 + box_h;
        cv::Point p1(x0, y0);
        cv::Point p2(x1, y1);
        cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2);
        std::stringstream ss;
        ss << round(result[j][4] * 1000) / 1000.0;
        int label = static_cast<int>(result[j][5]);
        if (label > 79 || label < 0) continue;
        std::string str =
            labelToDisplayName[static_cast<int>(result[j][5])] + ":"+ss.str();
        cv::Point p5(x0, y0 + 10);
        cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(240, 240, 0), 1);

        fileMap << labelToDisplayName[static_cast<int>(result[j][5])]
                << " " << ss.str()
                << " " << clip(x0 / image.cols,0,1)
                << " " << clip(y0 / image.rows,0,1)
                << " " << clip(x1 / image.cols,0,1)
                << " " << clip(y1 / image.rows,0,1)
                << " " << image.cols
                << " " << image.rows << std::endl;
      }
      fileMap.close();
      std::stringstream ss;
      std::string outFile;
      ss << FLAGS_outputdir << "/yolov8_offline_" << name << ".jpg";
      ss >> outFile;
      cv::imwrite((outFile.c_str()), image);
  }
}

INSTANTIATE_ALL_CLASS(YoloV8Processor);
#endif
