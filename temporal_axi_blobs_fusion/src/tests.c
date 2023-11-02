#include "tests.h"

void fill_merged_detection(struct single_axis_detection *azimuth, struct single_axis_detection *elevation,
						   struct merged_detection *merged_detection, struct imu_and_gps_data *imu_gps)
{
	// header
	memcpy(merged_detection->header.magic_CN, azimuth->header.magic_CN, 4);
	merged_detection->header.scan_number = azimuth->header.scan_number;
	merged_detection->header.ramp_number = azimuth->header.ramp_number;
	merged_detection->header.timestamp = azimuth->header.timestamp;
	merged_detection->header.processing_step = azimuth->header.processing_step;
	merged_detection->header.axis = 2;
	merged_detection->header.doppler_signature_cells = azimuth->header.doppler_signature_cells;
	memcpy(merged_detection->header.spare, azimuth->header.spare, 8);

	// data
	merged_detection->data.azimuth = azimuth->data.azimuth_elevation;
	merged_detection->data.elevation = elevation->data.azimuth_elevation;
	merged_detection->data.distance = (azimuth->data.distance + elevation->data.distance) / 2;
	merged_detection->data.min_azimuth = azimuth->data.min_azimuth_elevation;
	merged_detection->data.min_elevation = elevation->data.min_azimuth_elevation;
	merged_detection->data.max_elevation = elevation->data.max_azimuth_elevation;
	merged_detection->data.max_azimuth = azimuth->data.max_azimuth_elevation;
	merged_detection->data.min_distance = azimuth->data.min_distance;
	merged_detection->data.max_distance = azimuth->data.max_distance;
	merged_detection->data.pos_axis_x = 34;
	merged_detection->data.pos_axis_y = 35;
	merged_detection->data.pos_axis_z = 36;
	merged_detection->data.monopulse_ratio_az = azimuth->data.monopulse_ratio;
	merged_detection->data.monopulse_ratio_el = elevation->data.monopulse_ratio;


	memcpy(merged_detection->doppler_signature_azimuth, azimuth->data.doppler_signature,
		   sizeof(float) * 2 * NUM_MAX_DETECTIONS);
	memcpy(merged_detection->doppler_signature_elevation, elevation->data.doppler_signature,
		   sizeof(float) * 2 * NUM_MAX_DETECTIONS);
	memcpy(&(merged_detection->imu_gps), imu_gps, sizeof(struct imu_and_gps_data));
}

void execute_axis_blobs_fusion_cpu(struct single_axis_detection *elevation,
								   struct single_axis_detection *azimuth,
								   struct merged_detection *merged_detection,
								   struct imu_and_gps_data *imu_gps, uint32_t detections_azimuth,
								   uint32_t detections_elevation)
{
	float dif_distance;
	uint32_t found = 0;
	uint32_t ramp_N_azimuth;
	uint32_t ramp_N_elevation;
	float abs;
	clock_t start = clock();
	for (uint32_t i = 0; i < detections_azimuth; i++) {
		found = 0;
		dif_distance = 456468;
		ramp_N_azimuth = azimuth[i].header.ramp_number;
		for (uint32_t j = 0; j < detections_elevation; j++) {
			ramp_N_elevation = elevation[j].header.ramp_number;
			abs = sqrt((azimuth[i].data.distance - elevation[j].data.distance)
					   * (azimuth[i].data.distance - elevation[j].data.distance));
			if (azimuth[i].data.distance + DISTANCE_WINDOW > elevation[j].data.distance
				&& azimuth[i].data.distance - DISTANCE_WINDOW < elevation[j].data.distance
				&& azimuth[i].data.doppler_signature[ramp_N_azimuth][REAL] + DOPPLER_WINDOW
					   > elevation[j].data.doppler_signature[ramp_N_elevation][REAL]
				&& azimuth[i].data.doppler_signature[ramp_N_azimuth][REAL] - DOPPLER_WINDOW
					   < elevation[j].data.doppler_signature[ramp_N_elevation][REAL]
				&& azimuth[i].data.doppler_signature[ramp_N_azimuth][IMAG] + DOPPLER_WINDOW
					   > elevation[j].data.doppler_signature[ramp_N_elevation][IMAG]
				&& azimuth[i].data.doppler_signature[ramp_N_azimuth][IMAG] - DOPPLER_WINDOW
					   < elevation[j].data.doppler_signature[ramp_N_elevation][IMAG]
				&& azimuth[i].header.scan_number == elevation[j].header.scan_number) {

				if (dif_distance > abs) {
					dif_distance = abs;
					fill_merged_detection(&azimuth[i], &elevation[j], &merged_detection[i], imu_gps);
				}
				found = 1;
			}
		}
		if (!found) {
			merged_detection[i].data.distance = -1;
		}
	}
	clock_t end = clock();

	printf("Time CPU %fs\n", (float) (end - start) / CLOCKS_PER_SEC);
}
