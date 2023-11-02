cmake_minimum_required(VERSION 3.0)
add_definitions(-DCL_TARGET_OPENCL_VERSION=300)
project(Proyect_axis_blobs_fusion)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format-truncation -mcmodel=large")


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/CMake)

include(axi_blobs_fusion)
