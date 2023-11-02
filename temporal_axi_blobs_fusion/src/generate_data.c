#include "generate_data.h"


void fill_data(struct single_axis_detection *azimuth, struct single_axis_detection *elevation,
			   struct imu_and_gps_data *imu_gps, int aletority_factor)
{

	for (int i = 0; i < NUM_MAX_DETECTIONS; i++) {
		azimuth[i].header.axis = rand() % 255;
		azimuth[i].header.doppler_signature_cells = rand() % 255;
		azimuth[i].header.magic_CN[0] = rand() % 255;
		azimuth[i].header.processing_step = rand() % 255;
		azimuth[i].header.ramp_number = 1;
		azimuth[i].header.scan_number = rand() / RAND_MAX + rand() % aletority_factor;
		azimuth[i].header.spare[0] = rand() % 255;
		azimuth[i].header.timestamp = rand() % 255;


		azimuth[i].data.azimuth_elevation = rand();
		azimuth[i].data.distance = rand() / RAND_MAX + rand() % aletority_factor;
		for (int j = 0; j < PAR_PROCESSING_NUM_RAMPS; j++) {
			azimuth[i].data.doppler_signature[j][0] = rand();
			azimuth[i].data.doppler_signature[j][1] = rand();
			azimuth[i].data.doppler_signature[azimuth[60].header.ramp_number][REAL]
				= rand() / RAND_MAX + rand() % aletority_factor;
			azimuth[i].data.doppler_signature[azimuth[60].header.ramp_number][IMAG]
				= rand() / RAND_MAX + rand() % aletority_factor;
		}
		azimuth[i].data.max_azimuth_elevation = rand();
		azimuth[i].data.max_distance = rand();
		azimuth[i].data.min_azimuth_elevation = rand();
		azimuth[i].data.min_distance = rand();
		azimuth[i].data.monopulse_ratio = rand();

		elevation[i].header.axis = rand() % 255;
		elevation[i].header.doppler_signature_cells = rand() % 255;
		elevation[i].header.magic_CN[0] = 'C';
		elevation[i].header.magic_CN[1] = 'N';
		elevation[i].header.magic_CN[2] = 'C';
		elevation[i].header.magic_CN[3] = 'N';
		elevation[i].header.processing_step = rand() % 255;
		elevation[i].header.ramp_number = 1;
		elevation[i].header.scan_number = rand() / RAND_MAX + rand() % aletority_factor;
		elevation[i].header.spare[0] = 69;
		elevation[i].header.timestamp = rand() % 255;


		elevation[i].data.azimuth_elevation = rand();
		elevation[i].data.distance = rand() / RAND_MAX + rand() % aletority_factor;
		for (int j = 0; j < PAR_PROCESSING_NUM_RAMPS; j++) {
			elevation[i].data.doppler_signature[j][0] = rand();
			elevation[i].data.doppler_signature[j][1] = rand();
			elevation[i].data.doppler_signature[elevation[60].header.ramp_number][REAL]
				= rand() / RAND_MAX + rand() % aletority_factor;
			elevation[i].data.doppler_signature[elevation[60].header.ramp_number][IMAG]
				= rand() / RAND_MAX + rand() % aletority_factor;
		}
		elevation[i].data.max_azimuth_elevation = rand();
		elevation[i].data.max_distance = rand();
		elevation[i].data.min_azimuth_elevation = rand();
		elevation[i].data.min_distance = rand();
		elevation[i].data.monopulse_ratio = rand();
	}

	imu_gps->time = 123456789;
	imu_gps->latitude = 28.28;
	imu_gps->longitude = 26.26;
	imu_gps->altitude = 21.21;
	imu_gps->ecef_position_x = 128.128;
	imu_gps->ecef_position_y = 228.228;
	imu_gps->ecef_position_z = 328.328;
	imu_gps->yaw = 2878.2878;
	imu_gps->pitch = 2845.2845;
	imu_gps->roll = 2812.2812;
	imu_gps->velocity_north = 1.1;
	imu_gps->velocity_east = 1.2;
	imu_gps->velocity_up = 1.3;
	imu_gps->acceleration_x = 1.4;
	imu_gps->acceleration_y = 1.5;
	imu_gps->acceleration_z = 1.6;


	// known values

	azimuth[6].header.scan_number = 10;
	azimuth[6].header.ramp_number = 1;
	azimuth[6].data.distance = 100.4;
	azimuth[6].data.doppler_signature[azimuth[6].header.ramp_number][REAL] = 10.46;
	azimuth[6].data.doppler_signature[azimuth[6].header.ramp_number][IMAG] = 20.46;

	elevation[60].header.scan_number = 10;
	elevation[60].header.ramp_number = 1;
	elevation[60].data.distance = 101.3;
	elevation[60].data.doppler_signature[elevation[60].header.ramp_number][REAL] = 10.46;
	elevation[60].data.doppler_signature[elevation[60].header.ramp_number][IMAG] = 20.46;


	elevation[76].header.scan_number = 10;
	elevation[76].header.ramp_number = 10;
	elevation[76].data.distance = 100.9;
	elevation[76].data.doppler_signature[elevation[76].header.ramp_number][REAL] = 10.56;
	elevation[76].data.doppler_signature[elevation[76].header.ramp_number][IMAG] = 20.36;


	elevation[86].header.scan_number = 10;
	elevation[86].header.ramp_number = 17;
	elevation[86].data.distance = 101.1;
	elevation[86].data.doppler_signature[elevation[86].header.ramp_number][REAL] = 10.66;
	elevation[86].data.doppler_signature[elevation[86].header.ramp_number][IMAG] = 20.26;


	elevation[88].header.scan_number = 10;
	elevation[88].header.ramp_number = 27;
	elevation[88].data.distance = 100.3;
	elevation[88].data.doppler_signature[elevation[88].header.ramp_number][REAL] = 10.77;
	elevation[88].data.doppler_signature[elevation[88].header.ramp_number][IMAG] = 20.77;


	azimuth[34].header.scan_number = 10;
	azimuth[34].header.ramp_number = 10;
	azimuth[34].data.distance = 200.4;
	azimuth[34].data.doppler_signature[azimuth[34].header.ramp_number][REAL] = 10.46;
	azimuth[34].data.doppler_signature[azimuth[34].header.ramp_number][IMAG] = 20.46;

	elevation[24].header.scan_number = 10;
	elevation[24].header.ramp_number = 10;
	elevation[24].data.distance = 200.3;
	elevation[24].data.doppler_signature[elevation[24].header.ramp_number][REAL] = 10.46;
	elevation[24].data.doppler_signature[elevation[24].header.ramp_number][IMAG] = 20.46;


	azimuth[44].header.scan_number = 10;
	azimuth[44].header.ramp_number = 10;
	azimuth[44].data.distance = 300.4;
	azimuth[44].data.doppler_signature[azimuth[44].header.ramp_number][REAL] = 10.46;
	azimuth[44].data.doppler_signature[azimuth[44].header.ramp_number][IMAG] = 20.46;

	elevation[78].header.scan_number = 10;
	elevation[78].header.ramp_number = 10;
	elevation[78].data.distance = 300.3;
	elevation[78].data.doppler_signature[elevation[78].header.ramp_number][REAL] = 10.46;
	elevation[78].data.doppler_signature[elevation[78].header.ramp_number][IMAG] = 20.46;
}
