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

#define DATA_SIZE (1024*512*1024)

////////////////////////////////////////////////////////////////////////////////

const char *kernel_sum = "\n" \
"__kernel void sum(                                                  	\n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int DATA_SIZE)                                       \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < DATA_SIZE)                                                   \n" \
"       output[i] = input[i] + input[i];                                \n" \
"	if(i==0){                                                           \n" \
"  		printf(\"sum input[0] = %f\\n\",input[0]);                      \n" \
"  		printf(\"sum output[0] = %f\\n\",output[0]);                    \n" \
"	}                                                                   \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

const char *kernel_square = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int DATA_SIZE)                                       \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < DATA_SIZE)                                                   \n" \
"       output[i] = input[i] * input[i];                                \n" \
"	if(i==0){                                                           \n" \
"  		printf(\"square input[0] = %f\\n\",input[0]);                   \n" \
"  		printf(\"square output[0] = %f\\n\",output[0]);                 \n" \
"	}                                                                   \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

struct gpu_square
{
	size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
	cl_platform_id platforms;			// platfor to work with
};	struct gpu_square square ;

struct gpu_sum
{
	size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    
	cl_platform_id platforms;			// platfor to work with
};	struct gpu_sum sum ;

uint32_t init_square(float * results_ptr,float * data_ptr){

	uint32_t err;

    err = clGetPlatformIDs(1, &square.platforms, NULL);
    err = clGetDeviceIDs(square.platforms, CL_DEVICE_TYPE_GPU, 1, &square.device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    square.context = clCreateContext(0, 1, &square.device_id, NULL, NULL, &err);
    if (!square.context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

	cl_command_queue_properties properties = 0; // Puedes especificar propiedades adicionales aquí si es necesario
	square.commands = clCreateCommandQueueWithProperties(square.context, square.device_id, &properties, &err);
    if (!square.commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    square.program = clCreateProgramWithSource(square.context, 1, (const char **) & kernel_square, NULL, &err);
    if (!square.program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    err = clBuildProgram(square.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
		
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(square.program, square.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); // debugging tool
        printf("%s\n", buffer);
        exit(1);
    }

    square.kernel = clCreateKernel(square.program, "square", &err);
    if (!square.kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Crear un array de entrada estando alojado en la cpu  CL_MEM_USE_HOST_PTR, data_ptr
    square.input = clCreateBuffer(square.context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(float) * DATA_SIZE, data_ptr, NULL);
	// Crear un array de salida estando alojado en la cpu  CL_MEM_USE_HOST_PTR, results_ptr
    square.output = clCreateBuffer(square.context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, sizeof(float) * DATA_SIZE, results_ptr, NULL);
    if (!square.input || !square.output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
	// vincular la memoria de la gpu y cpu mapear
	data_ptr = (float *)clEnqueueMapBuffer(square.commands, square.input, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * DATA_SIZE, 0, NULL, NULL, NULL);
	results_ptr = (float *)clEnqueueMapBuffer(square.commands, square.output, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * DATA_SIZE, 0, NULL, NULL, NULL);

	return err;
}

uint32_t execute_square(){
	uint32_t err;

	err = 0;
	err  = clSetKernelArg(square.kernel, 0, sizeof(cl_mem), &square.input);
	err |= clSetKernelArg(square.kernel, 1, sizeof(cl_mem), &square.output);
	uint32_t dimnsion = DATA_SIZE;
	err |= clSetKernelArg(square.kernel, 2, sizeof(unsigned int), &dimnsion);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	err = clGetKernelWorkGroupInfo(square.kernel, square.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(square.local), &square.local, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}
	square.global = DATA_SIZE;
	err = clEnqueueNDRangeKernel(square.commands, square.kernel, 1, NULL, &square.global, &square.local, 0, NULL, NULL);
	if (err)
	{
		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}
	clFinish(square.commands);
	clFlush(square.commands);

	return err;
}

uint32_t release_square(float * results,float * data){
	
	uint32_t err;
	

	// demapear mapear la memoria de la gpu en la cpu
	clEnqueueUnmapMemObject(square.commands, square.output, results, 0, NULL, NULL); 
	clEnqueueUnmapMemObject(square.commands, square.input, data, 0, NULL, NULL); 

    // Shutdown and cleanup

    clReleaseMemObject(square.input);
    clReleaseMemObject(square.output);
    clReleaseProgram(square.program);
    clReleaseKernel(square.kernel);
    clReleaseCommandQueue(square.commands);
    clReleaseContext(square.context);

	return err;

}







uint32_t init_sum(float * results_ptr,float * data_ptr){

	uint32_t err;

    err = clGetPlatformIDs(1, &sum.platforms, NULL);
    err = clGetDeviceIDs(sum.platforms, CL_DEVICE_TYPE_GPU, 1, &sum.device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    sum.context = clCreateContext(0, 1, &sum.device_id, NULL, NULL, &err);
    if (!sum.context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

	cl_command_queue_properties properties = 0; // Puedes especificar propiedades adicionales aquí si es necesario
	sum.commands = clCreateCommandQueueWithProperties(sum.context, sum.device_id, &properties, &err);
    if (!sum.commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    sum.program = clCreateProgramWithSource(sum.context, 1, (const char **) & kernel_sum, NULL, &err);
    if (!sum.program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    err = clBuildProgram(sum.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
		
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(sum.program, sum.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); // debugging tool
        printf("%s\n", buffer);
        exit(1);
    }

    sum.kernel = clCreateKernel(sum.program, "sum", &err);
    if (!sum.kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Crear un array de entrada estando alojado en la cpu  CL_MEM_USE_HOST_PTR, data_ptr
    sum.input = clCreateBuffer(sum.context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(float) * DATA_SIZE, data_ptr, NULL);
	// Crear un array de salida estando alojado en la cpu  CL_MEM_USE_HOST_PTR, results_ptr
    sum.output = clCreateBuffer(sum.context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, sizeof(float) * DATA_SIZE, results_ptr, NULL);
    if (!sum.input || !sum.output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    
	// vincular la memoria de la gpu y cpu mapear
	data_ptr = (float *)clEnqueueMapBuffer(sum.commands, sum.input, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * DATA_SIZE, 0, NULL, NULL, NULL);
	results_ptr = (float *)clEnqueueMapBuffer(sum.commands, sum.output, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * DATA_SIZE, 0, NULL, NULL, NULL);

	return err;
}

uint32_t execute_sum(){
	uint32_t err;

	err = 0;
	err  = clSetKernelArg(sum.kernel, 0, sizeof(cl_mem), &sum.input);
	err |= clSetKernelArg(sum.kernel, 1, sizeof(cl_mem), &sum.output);
	uint32_t dimnsion = DATA_SIZE;
	err |= clSetKernelArg(sum.kernel, 2, sizeof(unsigned int), &dimnsion);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	err = clGetKernelWorkGroupInfo(sum.kernel, sum.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(sum.local), &sum.local, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}
	sum.global = DATA_SIZE;
	err = clEnqueueNDRangeKernel(sum.commands, sum.kernel, 1, NULL, &sum.global, &sum.local, 0, NULL, NULL);
	if (err)
	{
		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}
	clFinish(sum.commands);
	clFlush(sum.commands);

	return err;
}

uint32_t release_sum(float * results,float * data){
	
	uint32_t err;
	

	// demapear mapear la memoria de la gpu en la cpu
	clEnqueueUnmapMemObject(sum.commands, sum.output, results, 0, NULL, NULL); 
	clEnqueueUnmapMemObject(sum.commands, sum.input, data, 0, NULL, NULL); 

    // Shutdown and cleanup

    clReleaseMemObject(sum.input);
    clReleaseMemObject(sum.output);
    clReleaseProgram(sum.program);
    clReleaseKernel(sum.kernel);
    clReleaseCommandQueue(sum.commands);
    clReleaseContext(sum.context);

	return err;

}

float intern[DATA_SIZE] __attribute__((aligned(128))) = {0}; 

uint32_t init_gpu(float * results_ptr,float * data_ptr){
	uint32_t err;
	

	err = init_sum(&intern[0],data_ptr);

	err = init_square(results_ptr,&intern[0]);

	return err;
}


uint32_t release_gpu(float * results,float * data){
	uint32_t err;
	err = release_sum(intern,data);
	
	err = release_square(results,intern);

	return err;
}

uint32_t execute_gpu(){
	uint32_t err;

	err = execute_sum();

	err = execute_square();

	return err;
}

int main(int argc, char** argv)
{
    int err;                            
    unsigned int correct;               
	float data[DATA_SIZE] __attribute__((aligned(128))) = {0}; 		// necesario alinear el comienzo de la memoria en opencl formato
	float results[DATA_SIZE] __attribute__((aligned(128))) = {0};	// necesario alinear el comienzo de la memoria en opencl formato
	
	printf("TEST\n");

	init_gpu(&results[0],&data[0]);

	int ite = 0;
	while(1){		
		ite ++;
		for(int i = 0; i < DATA_SIZE; i++)
			data[i] = rand() / (float)RAND_MAX + ite;

		clock_t start = clock();

		execute_gpu();

		clock_t end = clock();

		printf("execute time %fs\n",(float)(end - start)/CLOCKS_PER_SEC);


		correct = 0;
		for(int i = 0; i < DATA_SIZE; i++){
			if(results[i] == (data[i] + data[i]) * (data[i] + data[i])  && results[i] != 0)
				correct++;
		}	

		printf("Computed '%d/%d' correct values!\n", correct, DATA_SIZE);
		sleep(1);
	}

	release_gpu(results,data);

    return 0;
}
