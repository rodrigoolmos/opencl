#include "axis_blobs_fusionner.h"


uint32_t execute_axis_blobs_fusionner(unsigned int number_detections_azimuth,
									  unsigned int number_detections_elevation)
{
	uint32_t err = 0;
	err = clSetKernelArg(axis_blobs_gpu_stuff.kernel, 0, sizeof(cl_mem), &axis_blobs_gpu_stuff.dev_azimuth);
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 1, sizeof(cl_mem),
						  &axis_blobs_gpu_stuff.dev_elevation);
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 2, sizeof(struct imu_and_gps_data), &imu_gps);
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 3, sizeof(cl_mem), &axis_blobs_gpu_stuff.dev_merged);
	unsigned int NUM_MAX_DETECTIONS_AZIMUTH = number_detections_azimuth;
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 4, sizeof(unsigned int), &NUM_MAX_DETECTIONS_AZIMUTH);
	unsigned int NUM_MAX_DETECTIONS_ELEVATION = number_detections_elevation;
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 5, sizeof(unsigned int),
						  &NUM_MAX_DETECTIONS_ELEVATION);
	unsigned int distance_window = DISTANCE_WINDOW;
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 6, sizeof(unsigned int), &distance_window);
	unsigned int doppler_window = DOPPLER_WINDOW;
	err |= clSetKernelArg(axis_blobs_gpu_stuff.kernel, 7, sizeof(unsigned int), &doppler_window);
	if (err != CL_SUCCESS) {

		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	err = clGetKernelWorkGroupInfo(axis_blobs_gpu_stuff.kernel, axis_blobs_gpu_stuff.device_id,
								   CL_KERNEL_WORK_GROUP_SIZE, sizeof(axis_blobs_gpu_stuff.local),
								   &axis_blobs_gpu_stuff.local, NULL);
	if (err != CL_SUCCESS) {

		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}

	axis_blobs_gpu_stuff.global = NUM_MAX_DETECTIONS;
	err = clEnqueueNDRangeKernel(axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.kernel, 1, NULL,
								 &axis_blobs_gpu_stuff.global, &axis_blobs_gpu_stuff.local, 0, NULL, NULL);
	if (err) {

		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}

	clFinish(axis_blobs_gpu_stuff.commands);
	clFlush(axis_blobs_gpu_stuff.commands);
}
