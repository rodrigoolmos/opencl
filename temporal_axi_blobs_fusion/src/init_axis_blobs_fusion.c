#include "axis_blobs_fusionner.h"
#include "merge_plot_kernel.h"


struct single_axis_detection azimuth[NUM_MAX_DETECTIONS] __attribute__((aligned(256))) = {0};
struct single_axis_detection *azimuth_ptr = azimuth;
struct imu_and_gps_data imu_gps __attribute__((aligned(256))) = {0};
struct single_axis_detection elevation[NUM_MAX_DETECTIONS] __attribute__((aligned(256))) = {0};
struct single_axis_detection *elevation_ptr = elevation;
struct merged_detection merged[NUM_MAX_DETECTIONS] __attribute__((aligned(256))) = {0};
struct merged_detection *merged_ptr = merged;
struct axis_blobs_gpu_stuff axis_blobs_gpu_stuff;


uint32_t init_axis_blobs_fusion()
{

	int err;	// error code returned from api calls


	///////////////////////////////////////////////////////////////////////////////
	err = clGetPlatformIDs(1, &axis_blobs_gpu_stuff.platforms, NULL);
	err = clGetDeviceIDs(axis_blobs_gpu_stuff.platforms, CL_DEVICE_TYPE_GPU, 1,
						 &axis_blobs_gpu_stuff.device_id, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	axis_blobs_gpu_stuff.context = clCreateContext(0, 1, &axis_blobs_gpu_stuff.device_id, NULL, NULL, &err);
	if (!axis_blobs_gpu_stuff.context) {
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	axis_blobs_gpu_stuff.commands
		= clCreateCommandQueue(axis_blobs_gpu_stuff.context, axis_blobs_gpu_stuff.device_id, 0, &err);
	if (!axis_blobs_gpu_stuff.commands) {
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	axis_blobs_gpu_stuff.program = clCreateProgramWithSource(axis_blobs_gpu_stuff.context, 1,
															 (const char **) &KernelSource, NULL, &err);
	if (!axis_blobs_gpu_stuff.program) {
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	err = clBuildProgram(axis_blobs_gpu_stuff.program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(axis_blobs_gpu_stuff.program, axis_blobs_gpu_stuff.device_id,
							  CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);	  // debugging tool
		printf("%s\n", buffer);
		exit(1);
	}

	axis_blobs_gpu_stuff.kernel
		= clCreateKernel(axis_blobs_gpu_stuff.program, "merged_detection_kernel", &err);
	if (!axis_blobs_gpu_stuff.kernel || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Crear un array de entrada estando alojado en la cpu  CL_MEM_USE_HOST_PTR, data_ptr
	axis_blobs_gpu_stuff.dev_azimuth
		= clCreateBuffer(axis_blobs_gpu_stuff.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						 sizeof(struct single_axis_detection) * NUM_MAX_DETECTIONS, azimuth_ptr, NULL);
	/*
		axis_blobs_gpu_stuff.dev_imu_gps
			= clCreateBuffer(axis_blobs_gpu_stuff.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
							 sizeof(struct imu_and_gps_data), imu_gps_ptr, NULL);
	*/
	axis_blobs_gpu_stuff.dev_elevation
		= clCreateBuffer(axis_blobs_gpu_stuff.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						 sizeof(struct single_axis_detection) * NUM_MAX_DETECTIONS, elevation_ptr, NULL);

	// Crear un array de salida estando alojado en la cpu  CL_MEM_USE_HOST_PTR, merged_ptr
	axis_blobs_gpu_stuff.dev_merged
		= clCreateBuffer(axis_blobs_gpu_stuff.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
						 sizeof(struct merged_detection) * NUM_MAX_DETECTIONS, merged_ptr, NULL);
	if (!axis_blobs_gpu_stuff.dev_azimuth || !axis_blobs_gpu_stuff.dev_elevation
		|| !axis_blobs_gpu_stuff.dev_merged /*|| !axis_blobs_gpu_stuff.dev_imu_gps*/) {
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}

	azimuth_ptr = (struct single_axis_detection *) clEnqueueMapBuffer(
		axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_azimuth, CL_TRUE, CL_MAP_WRITE, 0,
		sizeof(struct single_axis_detection) * NUM_MAX_DETECTIONS, 0, NULL, NULL, NULL);
	/*
		imu_gps_ptr = (struct imu_and_gps_data *) clEnqueueMapBuffer(
			axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_imu_gps, CL_TRUE, CL_MAP_WRITE, 0,
			sizeof(struct imu_and_gps_data), 0, NULL, NULL, NULL);
	*/
	elevation_ptr = (struct single_axis_detection *) clEnqueueMapBuffer(
		axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_elevation, CL_TRUE, CL_MAP_WRITE, 0,
		sizeof(struct single_axis_detection) * NUM_MAX_DETECTIONS, 0, NULL, NULL, NULL);
	merged_ptr = (struct merged_detection *) clEnqueueMapBuffer(
		axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_merged, CL_TRUE, CL_MAP_READ, 0,
		sizeof(struct merged_detection) * NUM_MAX_DETECTIONS, 0, NULL, NULL, NULL);
}
