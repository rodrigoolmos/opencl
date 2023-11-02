set(CMAKE_C_STANDARD 11)

set(AXI_BLOBS_FUSION_PATH ${CMAKE_CURRENT_LIST_DIR}/../)
message("Test")

file(READ  ${AXI_BLOBS_FUSION_PATH}/src/kernel.cl MERGE_PLOT_KERNEL_AUX1)
string(REPLACE "\"" "\\\"" MERGE_PLOT_KERNEL_AUX "${MERGE_PLOT_KERNEL_AUX1}")
string(REPLACE "\n" "\"\n\"" MERGE_PLOT_KERNEL "${MERGE_PLOT_KERNEL_AUX}")

configure_file(${AXI_BLOBS_FUSION_PATH}/autogen_kernels/merge_plot_kernel.h.in ${AXI_BLOBS_FUSION_PATH}/headers/merge_plot_kernel.h)

add_executable(axis_blobs_fusion main.c)

target_include_directories(axis_blobs_fusion PRIVATE ${AXI_BLOBS_FUSION_PATH}/headers)

file(GLOB SRC_FILES ${AXI_BLOBS_FUSION_PATH}/src/*.c)

target_sources(axis_blobs_fusion PRIVATE ${SRC_FILES})

set_target_properties(axis_blobs_fusion PROPERTIES COMPILE_FLAGS "-Dalignas=4096")

target_link_libraries(axis_blobs_fusion 
PUBLIC ${OpenCL_LIBRARY} OpenCL m)
