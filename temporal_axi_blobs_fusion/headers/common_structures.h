#ifndef COMMON_STRUCTURES
#define COMMON_STRUCTURES

#include <stdint.h>


#define NUM_MAX_DETECTIONS (64 * 400)


#define DISTANCE_WINDOW 1
#define DOPPLER_WINDOW 1

#define PAR_PROCESSING_NUM_RAMPS 1024

#define REAL 0
#define IMAG 1


#pragma pack(push)

/* detection_header structure ------------------------------------------- */

/**
   @struct       detection_header
   @brief        This structure contains the header used by single_axis_detection and merged_detection

   @var detection_header::magic_CN
   Data synchronization

   @var detection_header::scan_number
   Scan number correspondant to the Coherent Interval Processing

   @var detection_header::ramp_number
   Ramp number

   @var detection_header::timestamp
   Unix timestamp

   @var detection_header::processing_step
   Level of processing stages

   @var detection_header::axis
   0: Azimuth	1: Elevation	2: Merged

   @var detection_header::doppler_signature_cells
   Number of cells of the doppler signature for each axis

   @var detection_header::spare
   Reserved

 */
struct __attribute__((packed)) detection_header {
	char magic_CN[4];
	uint32_t scan_number;
	uint32_t ramp_number;
	uint64_t timestamp;
	uint8_t processing_step;
	uint8_t axis;
	uint16_t doppler_signature_cells;
	uint8_t spare[8];
};

/* single_axis_detection_data structure ------------------------------------------- */

/**
   @struct       single_axis_detection_data
   @brief        This structure contains the data used by single_axis_detection

   @var single_axis_detection_data::azimuth_elevation
   Azimuth/Elevation in degrees from the radar to the detection

   @var single_axis_detection_data::distance
   Distance in meters from the radar to the detection

   @var single_axis_detection_data::min_azimuth_elevation
   Minimum azimuth/elevation of the detection in degrees

   @var single_axis_detection_data::max_azimuth_elevation
   Maximum azimuth/elevation of the detection in degrees

   @var single_axis_detection_data::min_distance
   Minimum range of the detection in meters

   @var single_axis_detection_data::max_distance
   Maximum range of the detection in meters

   @var single_axis_detection_data::monopulse_ratio
   Monopulse ration of the detection in the axis

   @var single_axis_detection_data::doppler_signature
   Doppler signature

 */

struct __attribute__((packed)) single_axis_detection_data {
	float azimuth_elevation;
	float distance;
	float min_azimuth_elevation;
	float max_azimuth_elevation;
	float min_distance;
	float max_distance;
	float monopulse_ratio;
	float doppler_signature[PAR_PROCESSING_NUM_RAMPS][2];
};


/* single_axis_detection structure ------------------------------------------- */

/**
   @struct       single_axis_detection
   @brief        This structure contains the information of a single axis detection

   @var single_axis_detection::header
   Detection header structure

   @var single_axis_detection::data
   Single axis detection data
 */
struct __attribute__((packed)) single_axis_detection {
	struct detection_header header;
	struct single_axis_detection_data data;
};


/**
   @struct       imu_and_gps_data
   @brief        This structure contains the information of the IMU and the GPS

   @var imu_and_gps_data::time
   UTC Time stamp provided by IMU unit (ns)

   @var imu_and_gps_data::latitude
   Latitude position information of WG84 coordinate system (degrees)

   @var imu_and_gps_data::longitude
   Longitude position information of WG84 coordinate system (degrees)

   @var imu_and_gps_data::altitude
   Geodetic altitude of WG84 coordinate systems (meters)

   @var imu_and_gps_data::ecef_position_x
   Component X of position information given in ECEF coordinates (m)

   @var imu_and_gps_data::ecef_position_y
   Component Y of position information given in ECEF coordinates (m)

   @var imu_and_gps_data::ecef_position_z
   Number of cells of the doppler signature for each axis

   @var imu_and_gps_data::yaw
   Heading/yaw angle of the vehicle around Z axis. (degrees)

   @var imu_and_gps_data::pitch
   Pitch angle of the vehicle around Y axis. (degrees)

   @var imu_and_gps_data::roll
   Roll angle of the vehicle around X axis. degrees)

   @var imu_and_gps_data::velocity_north
   Velocity North in local ENU coordinate system of device. (m/s)

   @var imu_and_gps_data::velocity_east
   Velocity East in local ENU coordinate system of device. (m/s)

   @var imu_and_gps_data::velocity_up
   Velocity Down in local ENU coordinate system of device. (m/s)

   @var imu_and_gps_data::acceleration_x
   Estimated acceleration in the body frame (m/s2)

   @var imu_and_gps_data::acceleration_y
   Estimated acceleration in the body frame (m/s2)

   @var imu_and_gps_data::acceleration_z
   Estimated acceleration in the body frame (m/s2)

 */


struct __attribute__((packed)) imu_and_gps_data {
	uint64_t time;
	double latitude;
	double longitude;
	double altitude;
	double ecef_position_x;
	double ecef_position_y;
	double ecef_position_z;
	float yaw;
	float pitch;
	float roll;
	float velocity_north;
	float velocity_east;
	float velocity_up;
	float acceleration_x;
	float acceleration_y;
	float acceleration_z;
};

/* merged_detection_data structure ------------------------------------------- */

/**
   @struct       merged_detection_data
   @brief        This structure contains the data used by merged_detection

   @var merged_detection_data::azimuth
   Azimuth in degrees from the radar to the detection

   @var merged_detection_data::elevation
   Elevation in degrees from the radar to the detection

   @var merged_detection_data::distance
   Distance in meters from the radar to the detection

   @var merged_detection_data::min_azimuth
   Minimum azimuth of the detection in degrees

   @var merged_detection_data::max_azimuth
   Maximum azimuth of the detection in degrees

   @var merged_detection_data::min_elevation
   Minimum elevation of the detection in degrees

   @var merged_detection_data::max_elevation
   Maximum elevation of the detection in degrees

   @var merged_detection_data::min_distance
   Minimum range of the detection in meters

   @var merged_detection_data::max_distance
   Maximum range of the detection in meters

   @var merged_detection_data::pos_axis_x
   X coordinate of the detection position in radar local ENU in meters

   @var merged_detection_data::pos_axis_y
   Y coordinate of the detection position in radar local ENU in meters

   @var merged_detection_data::pos_axis_z
   Z coordinate of the detection position in radar local ENU in meters

   @var merged_detection_data::monopulse_ratio_az
   Monopulse ration of the detection in the azimuth axis

   @var merged_detection_data::monopulse_ratio_el
   Monopulse ration of the detection in the elevation axis

 */

struct __attribute__((packed)) merged_detection_data {
	float azimuth;
	float elevation;
	float distance;
	float min_azimuth;
	float min_elevation;
	float max_elevation;
	float max_azimuth;
	float min_distance;
	float max_distance;
	float pos_axis_x;
	float pos_axis_y;
	float pos_axis_z;
	float monopulse_ratio_az;
	float monopulse_ratio_el;
};

/* single_axis_detection structure ------------------------------------------- */

/**
   @struct       merged_detection
   @brief        This structure contains the information of a merged axis detection

   @var merged_detection::header
   Detection header structure

   @var merged_detection::data
   Merged detection data

   @var merged_detection::imu_gps
   Imu and gps data structure

   @var merged_detection::doppler_signature
   Doppler signature
 */
struct __attribute__((packed)) merged_detection {
	struct detection_header header;
	struct merged_detection_data data;	  // operaciones mio
	struct imu_and_gps_data imu_gps;
	float doppler_signature_azimuth[PAR_PROCESSING_NUM_RAMPS][2];
	float doppler_signature_elevation[PAR_PROCESSING_NUM_RAMPS][2];
};

#pragma pack(pop) /* restore original alignment from stack */

#endif
