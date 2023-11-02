#ifndef AXI_BLOBS_FUSIONNER
#define AXI_BLOBS_FUSIONNER


#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include "common_structures.h"


struct axis_blobs_gpu_stuff {
	size_t global;
	size_t local;

	cl_device_id device_id;
	cl_context context;
	cl_command_queue commands;
	cl_program program;
	cl_kernel kernel;

	cl_mem dev_azimuth;
	// cl_mem dev_imu_gps;
	cl_mem dev_elevation;
	cl_mem dev_merged;

	cl_platform_id platforms;
};


extern struct single_axis_detection azimuth[NUM_MAX_DETECTIONS];
extern struct single_axis_detection *azimuth_ptr;
extern struct imu_and_gps_data imu_gps;
// extern struct imu_and_gps_data *imu_gps_ptr;
extern struct single_axis_detection elevation[NUM_MAX_DETECTIONS];
extern struct single_axis_detection *elevation_ptr;
extern struct merged_detection merged[NUM_MAX_DETECTIONS];
extern struct merged_detection *merged_ptr;

extern const char *KernelSource;
extern struct axis_blobs_gpu_stuff axis_blobs_gpu_stuff;


uint32_t init_axis_blobs_fusion();

uint32_t execute_axis_blobs_fusionner(unsigned int number_detections_azimuth,
									  unsigned int number_detections_elevation);

uint32_t release_axis_blobs_fusion();

#endif
