#!/bin/bash

shopt -s nocasematch
EXAMPLE_DIR=$(dirname $(dirname $(readlink -f $0)))
PYTORCH_DIR=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))

if [ -z ${CROSS_TOOLCHAIN_PATH} ]; then
  echo "please set environment variable CROSS_TOOLCHAIN_PATH, export CROSS_TOOLCHAIN_PATH=your CROSS_TOOLCHAIN_PATH dir"
  exit 1
fi
export PATH=${CROSS_TOOLCHAIN_PATH}:$PATH
TOOLCHAIN_PREFIX="`echo ${CROSS_TOOLCHAIN_PATH%\-*}`-"
CROSS_COMPILE=aarch64-linux-gnu-

if [ -z ${AARCH64_LINUX_LIB_ROOT} ]; then
  echo "please set environment variable AARCH64_LINUX_LIB_ROOT, export AARCH64_LINUX_LIB_ROOT=your AARCH64_LINUX_LIB_ROOT dir"
  exit 1
fi
# export PROTOBUF_HOME="${AARCH64_LINUX_LIB_ROOT}/protobuf"
export GFLAGS_HOME="${AARCH64_LINUX_LIB_ROOT}/gflags"
export GLOG_HOME="${AARCH64_LINUX_LIB_ROOT}/glog"
export BOOST_HOME="${AARCH64_LINUX_LIB_ROOT}/boost"
# export OPENBLAS_HOME="${AARCH64_LINUX_LIB_ROOT}/openblas"
# export LMDB_HOME="${AARCH64_LINUX_LIB_ROOT}/lmdb"

BUILD_DIR=$EXAMPLE_DIR/"build"
if [ -z ${NEUWARE_HOME} ]; then
  echo "please set environment variable NEUWARE_HOME, export NEUWARE_HOME=your NEUWARE_HOME dir"
  exit 1
fi
# check build folder
if [ ! -d ${BUILD_DIR} ]; then
    echo "mkdir build folder"
    mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

BUILD_TYPE="Release"
if [[ $DEBUG == 1 ]]; then
  BUILD_TYPE="DEBUG"
fi


echo "=== cmake =============================================================="
cmake -DCROSS_COMPILE="${TOOLCHAIN_PREFIX}" \
  -DCMAKE_C_COMPILER="${CROSS_COMPILE}gcc" \
  -DCMAKE_CXX_COMPILER="${CROSS_COMPILE}g++" \
  -DCMAKE_CXX_FLAGS=" -fPIC -lgomp" \
  -DCMAKE_C_FLAGS=" -fPIC -lgomp" \
  -DUSE_MLU=ON \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DGFLAGS_INCLUDE_DIR="${GFLAGS_HOME}/include" \
  -DGFLAGS_LIBRARY="${GFLAGS_HOME}/lib/libgflags.so" \
  -DGLOG_INCLUDE_DIR="${GLOG_HOME}/include" \
  -DGLOG_LIBRARY="${GLOG_HOME}/lib/libglog.so" \
  -DBOOST_ROOT="${BOOST_HOME}" \
  -DCMAKE_INSTALL_PREFIX=../install \
  ..

make install -j16
echo "=== build =============================================================="
