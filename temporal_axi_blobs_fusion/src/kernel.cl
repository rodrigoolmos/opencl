struct __attribute__ ((packed)) doppler_signature															
{																											
	float doppler_signature[1024][2];																
};    																											
struct __attribute__ ((packed)) detection_header															
{																											
	char magic_CN[4];																						
	unsigned int scan_number;																				
	unsigned int ramp_number;																				
	unsigned long timestamp;																				
	char processing_step;																					
	char axis;																								
	unsigned short doppler_signature_cells;																	
	char spare[8];																							
};    																											
struct __attribute__ ((packed)) single_axis_detection_data													
{																											
	float azimuth_elevation;																				
	float distance;    																						
	float min_azimuth_elevation;    																		
	float max_azimuth_elevation;    																		
	float min_distance;    																					
	float max_distance;    																					
	float monopulse_ratio;																					
	struct doppler_signature doppler_signature;																
};    																											
struct __attribute__ ((packed)) single_axis_detection{														
																											
	struct detection_header header;																				
	struct single_axis_detection_data data;																	
};    
																											
struct __attribute__ ((packed)) imu_and_gps_data																
{																											
	unsigned long time;																						
	long latitude;																							
	long longitude;																							
	long altitude;																							
	long ecef_position_x;																					
	long ecef_position_y;																					
	long ecef_position_z;																					
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

struct __attribute__ ((packed)) merged_detection_data														
{																											
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

struct __attribute__ ((packed)) merged_detection															
{																											
	struct detection_header header;																			
	struct merged_detection_data data;																		
	struct imu_and_gps_data imu_gps;																		
	struct doppler_signature doppler_signature_azimuth;														
	struct doppler_signature doppler_signature_elevation;													
};     

__kernel void merged_detection_kernel(                                                  									
   	__global struct single_axis_detection* azimuth,                                        					
   	__global struct single_axis_detection* elevation,                                        				
   	const struct imu_and_gps_data imu_gps,                                       						
   	__global struct merged_detection* merged,                                       						
   	const unsigned int NUM_MAX_DETECTIONS_AZIMUTH,
   	const unsigned int NUM_MAX_DETECTIONS_ELEVATION,
   	const unsigned int DISTANCE_WINDOW,
   	const unsigned int DOPPLER_WINDOW){

   	int i = get_global_id(0);

   	float dif_distance_fusioned = 456468;
   	float distance_azimuth_tmp = azimuth[i].data.distance;
   	unsigned int ramp_N_azimuth = azimuth[i].header.ramp_number;
	float doppler_azimuth_tmp_real = 
		azimuth[i].data.doppler_signature.doppler_signature[ramp_N_azimuth][0];
   	float doppler_azimuth_tmp_img =  
		azimuth[i].data.doppler_signature.doppler_signature[ramp_N_azimuth][1];
	unsigned int ramp_N_elevation;
   	float doppler_elevation_tmp_real;                     
   	float doppler_elevation_tmp_img;                      
   	float distance_elevation_tmp;                         
   	float abs_value;

	unsigned int found = 0;                                      
	

   	if(i < NUM_MAX_DETECTIONS_AZIMUTH){   
		for (int j = 0; j < NUM_MAX_DETECTIONS_ELEVATION; j++){

			ramp_N_elevation = elevation[j].header.ramp_number;
			doppler_elevation_tmp_real = 
				elevation[j].data.doppler_signature.doppler_signature[ramp_N_elevation][0];                                                        
			doppler_elevation_tmp_img  = 
				elevation[j].data.doppler_signature.doppler_signature[ramp_N_elevation][1];                                                         
			distance_elevation_tmp = elevation[j].data.distance;                                                                
			abs_value = sqrt((distance_azimuth_tmp - distance_elevation_tmp)*
				(distance_azimuth_tmp - distance_elevation_tmp));
			if( distance_azimuth_tmp  + DISTANCE_WINDOW > distance_elevation_tmp &&			
 				distance_azimuth_tmp  - DISTANCE_WINDOW < distance_elevation_tmp && 		
				doppler_azimuth_tmp_real + DOPPLER_WINDOW > doppler_elevation_tmp_real &&	
 				doppler_azimuth_tmp_real - DOPPLER_WINDOW < doppler_elevation_tmp_real &&	
				doppler_azimuth_tmp_img + DOPPLER_WINDOW > doppler_elevation_tmp_img &&		
 				doppler_azimuth_tmp_img - DOPPLER_WINDOW < doppler_elevation_tmp_img &&		
				azimuth[i].header.scan_number == elevation[j].header.scan_number){
				if(dif_distance_fusioned > abs_value){
					dif_distance_fusioned = abs_value;
					
       				merged[i].header = azimuth[i].header;  
					merged[i].header.axis = 2;   
       				merged[i].data.azimuth = azimuth[i].data.azimuth_elevation;             							
       				merged[i].data.elevation = elevation[j].data.azimuth_elevation;             						
       				merged[i].data.distance = (distance_azimuth_tmp + distance_elevation_tmp)/2;             									
       				merged[i].data.min_azimuth = azimuth[i].data.min_azimuth_elevation;             					
       				merged[i].data.min_elevation = elevation[j].data.min_azimuth_elevation;             				
       				merged[i].data.max_elevation = elevation[j].data.max_azimuth_elevation;             				
       				merged[i].data.max_azimuth = azimuth[i].data.max_azimuth_elevation;             					
       				merged[i].data.min_distance = azimuth[i].data.min_distance;             							
       				merged[i].data.max_distance = azimuth[i].data.max_distance;             							
       				merged[i].data.pos_axis_x = 34;             														
       				merged[i].data.pos_axis_y = 35;             														
       				merged[i].data.pos_axis_z = 36;             														
       				merged[i].data.monopulse_ratio_az = azimuth[i].data.monopulse_ratio;             					
       				merged[i].data.monopulse_ratio_el = elevation[j].data.monopulse_ratio;  
       				merged[i].doppler_signature_azimuth = azimuth[i].data.doppler_signature;             				
       				merged[i].doppler_signature_elevation = elevation[j].data.doppler_signature;
					merged[i].imu_gps = imu_gps; 
					found = 1;
				}
			}
				
		}
		if(!found){
			merged[i].data.distance = -1;
		}             			
	}
	if(i==0){
		printf("DISTANCE_WINDOW = %i \\n",DISTANCE_WINDOW);
		printf("DOPPLER_WINDOW = %i \\n",DOPPLER_WINDOW);
		printf("NUM_MAX_DETECTIONS_AZIMUTH = %i \\n",NUM_MAX_DETECTIONS_AZIMUTH);
		printf("NUM_MAX_DETECTIONS_ELEVATION = %i \\n",NUM_MAX_DETECTIONS_ELEVATION);
		printf("KERNEL_WORK_GROUP_SIZE  %i \\n",get_global_size(0));

	}                                                                   									                                                                 									
}                                                                      									
