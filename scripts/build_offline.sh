#!/bin/bash
shopt -s nocasematch
EXAMPLE_DIR=$(dirname $(dirname $(readlink -f $0)))
PYTORCH_DIR=$(dirname $(dirname $(dirname $(dirname $(readlink -f $0)))))
BUILD_DIR=$EXAMPLE_DIR/"build"
BUILD_TYPE="Release"
# check build folder
if [ ! -d ${BUILD_DIR} ]; then
    echo "mkdir build folder"
    mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

if [[ $DEBUG == 1 ]]; then
  BUILD_TYPE="DEBUG"
fi

pushd $BUILD_DIR
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DCMAKE_SKIP_RPATH=TRUE \
      ..

# Be nice
make -j${MAX_JOBS}

if [ $? -ne 0 ]; then
    popd
    exit 1
fi

popd
