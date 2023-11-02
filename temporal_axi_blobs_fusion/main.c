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
#include "tests.h"
#include "generate_data.h"
#include "axis_blobs_fusionner.h"


int main()
{

	int err;	// error code returned from api calls

	uint32_t detections_azimuth = NUM_MAX_DETECTIONS;
	uint32_t detections_elevation = NUM_MAX_DETECTIONS;


	unsigned int correct = 0;	 // number of correct merged returned

	unsigned int cpu_detections = 0;	// number of correct merged returned

	unsigned int gpu_detections = 0;	// number of correct merged returned

	struct merged_detection merged_cpu[NUM_MAX_DETECTIONS] __attribute__((aligned(256))) = {0};
	struct merged_detection merged_result[NUM_MAX_DETECTIONS] __attribute__((aligned(256))) = {0};


	init_axis_blobs_fusion();


	int random_factor = 0;
	while (1) {

		correct = 0;
		cpu_detections = 0;
		gpu_detections = 0;

		detections_azimuth = rand() % NUM_MAX_DETECTIONS;
		detections_elevation = rand() % NUM_MAX_DETECTIONS;

		random_factor++;

		fill_data(azimuth, elevation, &imu_gps, random_factor);

		clock_t start = clock();

		execute_axis_blobs_fusionner(detections_azimuth, detections_elevation);

		clock_t end = clock();

		printf("Time GPU %fs\n", (float) (end - start) / CLOCKS_PER_SEC);

		printf("NUM_MAX_DETECTIONS  %i\n", NUM_MAX_DETECTIONS);

		execute_axis_blobs_fusion_cpu(elevation, azimuth, merged_cpu, &imu_gps, detections_azimuth,
									  detections_elevation);

		for (size_t i = 0; i < detections_azimuth; i++) {

			if (!memcmp(&merged_cpu[i], &merged[i], sizeof(struct merged_detection))
				& merged[i].data.distance >= 0) {
				correct++;
			} else {
				int test = 0;
			}
		}


		// relistic timing last part of fusionning azimuth elevation
		start = clock();
		for (int i = 0; i < detections_azimuth; i++) {
			if (merged_cpu[i].data.distance >= 0) {
				cpu_detections++;
			}

			if (merged[i].data.distance >= 0) {
				gpu_detections++;
			}
			memcpy(&merged_result[i], &merged[i], sizeof(struct merged_detection));
		}
		end = clock();

		printf("Time MEMCPY %fs\n", (float) (end - start) / CLOCKS_PER_SEC);

		printf("size %lu\n", sizeof(struct merged_detection));
		printf("Computed '%d/%d' correct values!\n", correct, gpu_detections);
		printf("CPU detections %i\n", cpu_detections);
		printf("GPU detections %i\n", gpu_detections);
		printf("Random factor %i\n", random_factor);


		sleep(1);
	}

	release_axis_blobs_fusion();

	return 0;
}
