cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Offline project
project(Offline C CXX)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules ${PROJECT_SOURCE_DIR}/../../cmake/modules)

include(ExternalProject)
include(GNUInstallDirs)
include(cmake/Utils.cmake)
#include(cmake/Targets.cmake)
include(cmake/Misc.cmake)

# ---[ Options
# offline_option(USE_MLU  "Build offline examples with CNRT support" ON)
# offline_option(CROSS_COMPILE  "don't use resize when build arm32" OFF)
# offline_option(CROSS_COMPILE_ARM64  "don't use resize when build arm64" OFF)
# offline_option(USE_OPENCV "Build with OpenCV support" ON)
# offline_option(NMS_ORDER "USE NMS WITH ORDER" ON)
OPTION(USE_MLU  "Build offline examples with CNRT support" ON)
OPTION(CROSS_COMPILE  "don't use resize when build arm32" OFF)
OPTION(CROSS_COMPILE_ARM64  "don't use resize when build arm64" ON)
OPTION(USE_OPENCV "Build with OpenCV support" ON)
OPTION(NMS_ORDER "USE NMS WITH ORDER" ON)

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()
if($ENV{USE_ABI} MATCHES "1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
elseif($ENV{USE_ABI} MATCHES "0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
else()
  message("-- Use default ABI settings")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -pthread -Wfatal-errors -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")    # off opt
set(CMAKE_BUILD_TYPE "Release") #debug

warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Includes
include_directories(${PROJECT_BINARY_DIR})

include_directories(${GFLAGS_INCLUDE_DIR})
include_directories(${GLOG_INCLUDE_DIR})

# opencv
set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rdparty/opencv/share/OpenCV) # 填入OpenCVConfig.cmake
find_package(OpenCV  REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

set(OPENCV_COMMON_DEPENDENCY core highgui imgproc)
list(APPEND INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
list(APPEND LINKER_LIBS PUBLIC ${OpenCV_LIBS})
message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
list(APPEND DEFINITIONS PUBLIC -DUSE_OPENCV)

# # Need to include cxx standard include dirs on arm64
# if(CROSS_COMPILE_ARM64)
#   include_directories("${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES}")
# endif()

set(ALL_DEFINITIONS ${DEFINITIONS})
list(REMOVE_ITEM ALL_DEFINITIONS PRIVATE PUBLIC)
add_definitions(${ALL_DEFINITIONS})

# ---[ Subdirectories
add_subdirectory(common)
add_subdirectory(common_yolov5)
add_subdirectory(yolov5)
add_subdirectory(yolov8)


install(DIRECTORY ${PROJECT_SOURCE_DIR}/3rdparty/ DESTINATION ./)
