#ifndef GENERATE_DATA
#define GENERATE_DATA


#include <stdio.h>
#include <stdlib.h>
#include "common_structures.h"


void fill_data(struct single_axis_detection *azimuth, struct single_axis_detection *elevation,
			   struct imu_and_gps_data *imu_gps, int ite);

#endif
