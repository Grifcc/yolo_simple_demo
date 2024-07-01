# Collect source files
file(GLOB OFFLINE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/off_*.cpp)
set(COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/blocking_queue.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/pipeline.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/data_provider.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/clas_processor.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/common_functions.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/handler.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/gflags_common.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/simple_interface.cpp"
                "${CMAKE_CURRENT_SOURCE_DIR}/runner_strategy.cpp")
set(COM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB RESIZE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/bicubicResize/*.cpp)
set(RESIZE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/bicubicResize/include")

include_directories(${CNRT_INCLUDE_DIRS} ${COM_INCLUDE_DIRS} ${INCLUDE_DIRS} ${RESIZE_INCLUDE_DIRS})

add_library(OffComLib STATIC ${OFFLINE_SRCS} ${COMMON_SRCS} ${RESIZE_SRCS})

target_link_libraries(OffComLib ${LINKER_LIBS})
