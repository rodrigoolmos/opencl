#include "axis_blobs_fusionner.h"

uint32_t release_axis_blobs_fusion()
{
	// demapear mapear la memoria de la gpu en la cpu
	clEnqueueUnmapMemObject(axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_merged, merged, 0, NULL,
							NULL);
	clEnqueueUnmapMemObject(axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_azimuth, azimuth, 0, NULL,
							NULL);
	// clEnqueueUnmapMemObject(axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_imu_gps, &imu_gps, 0,
	// 						NULL, NULL);
	clEnqueueUnmapMemObject(axis_blobs_gpu_stuff.commands, axis_blobs_gpu_stuff.dev_elevation, elevation, 0,
							NULL, NULL);

	// Shutdown and cleanup

	clReleaseMemObject(axis_blobs_gpu_stuff.dev_azimuth);
	// clReleaseMemObject(axis_blobs_gpu_stuff.dev_imu_gps);
	clReleaseMemObject(axis_blobs_gpu_stuff.dev_elevation);
	clReleaseMemObject(axis_blobs_gpu_stuff.dev_merged);
	clReleaseProgram(axis_blobs_gpu_stuff.program);
	clReleaseKernel(axis_blobs_gpu_stuff.kernel);
	clReleaseCommandQueue(axis_blobs_gpu_stuff.commands);
	clReleaseContext(axis_blobs_gpu_stuff.context);
}
