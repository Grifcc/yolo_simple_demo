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

#include <gflags/gflags.h>

DEFINE_string(meanfile, "", "mean file used to subtract from the input image.");
DEFINE_string(meanvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either meanfile or meanvalue should be provided, not both.");
DEFINE_string(stdfile, "", "std file used to subtract from the input image.");
DEFINE_string(stdvalue, "",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either stdfile or stdvalue should be provided, not both.");
DEFINE_int32(threads, 1, "threads should "
                         "be lower than or equal to 32 ");
DEFINE_int32(channel_dup, 1, "Enable const memory auto channel duplication. "
                         "Could improve performance when multithreading."
                         "Works only with apiversion 3");
DEFINE_int32(simple_compile, 1, "Use simple compile interface.");
DEFINE_string(images, "", "input file list");
DEFINE_string(dataset_path, "", "absolute dataset path");
DEFINE_string(labels, "", "label to name");
DEFINE_int32(fix8, 0, "fp16(0) or fix8(1) mode. Default is fp16");
DEFINE_int32(int8, -1, "invalid(-1), fp16(0) or int8(1) mode. Default is invalid(-1)."
    "If specified, use int8 value, else, use fix8 value");
DEFINE_int32(yuv, 0, "bgr(0) or yuv(1) mode. Default is bgr");
DEFINE_int32(rgb, 1, "bgr(0) or rgb(1) mode. Default is rgb");
DEFINE_int32(interpolation, 3, "opencv resize interpolation method"
                          "INTER_NEAREST(0), \
                           INTER_LINEAR(1), \
                           INTER_CUBIC(2), \
                           INTER_AREA(3), \
                           INTER_LANCZOS4(4), \
                           INTER_LINEAR_EXACT(5), \
                           INTER_NEAREST_EXACT(6), \
                           INTER_MAX(7), \
                           WARP_FILL_OUTLIERS(8), \
                           WARP_INVERSE_MAP(16). Default is INTER_AREA");
DEFINE_double(scale, 1, "scale for input data, mobilenet...");
DEFINE_string(logdir, "", "path to dump log file, to terminal stderr by default");
DEFINE_int32(fifosize, 2, "set FIFO size of mlu input and output buffer, default is 2");
DEFINE_string(mludevice, "0",
    "set using mlu device number, set multidevice seperated by ','"
    "eg 0,1 when you use device number 0 and 1, default: 0");
DEFINE_string(resize, "",
    "set preprocess resizes, set height,width seperated by ',' eg 256,256 "
    "when size is an int, the smaller edge is resize, "
    "the larger edge is (size * larger_edge / smaller_edge)");
DEFINE_int32(apiversion, 2, "specify the version of CNRT to run.");
DEFINE_string(functype, "1H16",
              "Specify the core to run on the arm device. "
              "Set the options to 1H16 or 1H8, the default is 1H16.");
DEFINE_int32(async_copy, 0,
             "Enable async copy, 0-disable 1-enable. Default is 0-disable.");
DEFINE_int32(input_format, 0, "input image channel order, default is 0(rgba)");
DEFINE_int32(dim_mutable, 0,
             "nchw is not mutable(0) or nchw is mutable(1). Default is not mutable(0)");
DEFINE_int32(dim_mutable_max_batch, 4,
             "when n is mutable, dim_mutable_max_batch is the maxsize of n ");
DEFINE_int32(timestamp, 0, "print pipeline timestamp");
DEFINE_string(offlinemodel, "", "prototxt file used to find net configuration");
DEFINE_int32(preprocess_method, 0, "Use it to choose Image preprocess");
// parameters for performance mode
DEFINE_int32(perf_mode, 0, "close(0) or open(1) performance mode, Default is close(0)");
DEFINE_int32(perf_mode_img_num, 4800, "number of images for performance mode, Default is 4800");


