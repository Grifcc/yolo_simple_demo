# Try to find the MLU libraries and headers

#find_library doesn't work while cross-compiling arm64 on dev branch
#although it worked while I compile with cnml on at branch
#right now we simplifying the library finding
SET(CNRT_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)
SET(CNRT_LIB_SEARCH_PATHS $ENV{NEUWARE_HOME}/lib64)

find_path(CNRT_INCLUDE_DIR NAMES cnrt.h
          PATHS ${CNRT_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(CNRT_INCLUDE_DIR NAMES cnrt.h
          NO_CMAKE_FIND_ROOT_PATH)

find_library(CNRT_LIBRARY NAMES cnrt
          PATHS ${CNRT_LIB_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_library(CNRT_LIBRARY NAMES cnrt
          NO_CMAKE_FIND_ROOT_PATH)

find_library(CNDRV_LIBRARY NAMES cndrv
          PATHS ${CNRT_LIB_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_library(CNDRV_LIBRARY NAMES cndrv
          NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLU DEFAULT_MSG CNRT_INCLUDE_DIR CNRT_LIBRARY)

if(MLU_FOUND)
  message(STATUS "Found CNRT (include: ${CNRT_INCLUDE_DIR}
     library: ${CNRT_LIBRARY})")
  parse_header(MLU_VERSION_LINES MLU_VERSION_MAJOR MLU_VERSION_MINOR MLU_VERSION_PATCH)
  set(MLU_VERSION "${MLU_VERSION_MAJOR}.${MLU_VERSION_MINOR}.${MLU_VERSION_PATCH}")
endif()
