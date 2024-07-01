set(LINKER_LIBS "")
set(INCLUDE_DIRS "")
set(DEFINITIONS "")
set(COMPILE_OPTIONS "")

# ---[ Boost
find_package(Boost 1.53 REQUIRED COMPONENTS system thread filesystem regex)
list(APPEND INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
list(APPEND LINKER_LIBS PUBLIC ${Boost_LIBRARIES})

# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ Google-glog
include("cmake/External/glog.cmake")
list(APPEND INCLUDE_DIRS PUBLIC ${GLOG_INCLUDE_DIRS})
list(APPEND LINKER_LIBS PUBLIC ${GLOG_LIBRARIES})

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
list(APPEND INCLUDE_DIRS PUBLIC ${GFLAGS_INCLUDE_DIRS})
list(APPEND LINKER_LIBS PUBLIC ${GFLAGS_LIBRARIES})

# ---[ MLU
if(USE_MLU)
    find_package(MLU REQUIRED)
  include_directories(SYSTEM ${CNRT_INCLUDE_DIR})
  list(APPEND LINKER_LIBS  PUBLIC ${CNRT_LIBRARY})
  list(APPEND LINKER_LIBS  PUBLIC ${CNDRV_LIBRARY})
  list(APPEND DEFINITIONS PUBLIC -DUSE_MLU)
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/scripts/install_githook.sh)
endif()

if(CROSS_COMPILE)
  list(APPEND DEFINITIONS PUBLIC -DCROSS_COMPILE)
endif()

if(CROSS_COMPILE_ARM64)
  list(APPEND DEFINITIONS PUBLIC -DCROSS_COMPILE_ARM64)
endif()



# ---[ OpenCV
# if(USE_OPENCV)
#   if(${CAMBRICOM_DRIVER_TYPE} MATCHES "mango_armv7")
#     set(OPENCV_COMMON_DEPENDENCY core highgui imgproc videoio)
#   else()
#     set(OPENCV_COMMON_DEPENDENCY core highgui imgproc)
#   endif()
#   find_package(OpenCV QUIET COMPONENTS ${OPENCV_COMMON_DEPENDENCY} imgcodecs)
#   if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
#     find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMMON_DEPENDENCY})
#   endif()
#   list(APPEND INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
#   list(APPEND LINKER_LIBS PUBLIC ${OpenCV_LIBS})
#   message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
#   list(APPEND DEFINITIONS PUBLIC -DUSE_OPENCV)
# endif()
