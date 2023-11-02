#ifndef TESTS
#define TESTS


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "common_structures.h"


void execute_axis_blobs_fusion_cpu(struct single_axis_detection *elevation,
								   struct single_axis_detection *azimuth,
								   struct merged_detection *merged_detection,
								   struct imu_and_gps_data *imu_gps, uint32_t detections_azimuth,
								   uint32_t detections_elevation);

#endif
