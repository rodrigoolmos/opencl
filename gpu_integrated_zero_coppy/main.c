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

const char *KernelSource = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int DATA_SIZE)                                       \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < DATA_SIZE)                                                   \n" \
"       output[i] = input[i] * input[i];                                \n" \
"	if(i==0){                                                           \n" \
"  		printf(\"input[0] = %f\\n\",input[0]);                        	\n" \
"  		printf(\"output[0] = %f\\n\",output[0]);                        	\n" \
"	}                                                                   \n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls
      
    unsigned int correct;               // number of correct results returned

	////////////////////////////////////////////////////////////////////////////
	//////////////////////     GPU stuff  ////////////////////////////////////// 

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
	////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////    CPU DATA   //////////////////////////////////
	float data[DATA_SIZE] __attribute__((aligned(64))) = {0}; // necesario alinear el comienzo de la memoria en opencl formato
	float *data_ptr = data;
	float results[DATA_SIZE] __attribute__((aligned(64))) = {0};
	float *results_ptr = results;

	///////////////////////////////////////////////////////////////////////////////
    err = clGetPlatformIDs(1, &platforms, NULL);
    err = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
		
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); // debugging tool
        printf("%s\n", buffer);
        exit(1);
    }

    kernel = clCreateKernel(program, "square", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Crear un array de entrada estando alojado en la cpu  CL_MEM_USE_HOST_PTR, data_ptr
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,  sizeof(float) * DATA_SIZE, data_ptr, NULL);
	// Crear un array de salida estando alojado en la cpu  CL_MEM_USE_HOST_PTR, results_ptr
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY| CL_MEM_USE_HOST_PTR, sizeof(float) * DATA_SIZE, results_ptr, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    

	data_ptr = (float *)clEnqueueMapBuffer(commands, input, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * DATA_SIZE, 0, NULL, NULL, NULL);
	results_ptr = (float *)clEnqueueMapBuffer(commands, output, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * DATA_SIZE, 0, NULL, NULL, NULL);

	int ite = 0;
	while(1){		
		ite ++;
		for(int i = 0; i < DATA_SIZE; i++)
			data[i] = rand() / (float)RAND_MAX + ite;

		clock_t start = clock();
		
		err = 0;
		err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
		uint32_t dimnsion = DATA_SIZE;
		err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &dimnsion);

		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to set kernel arguments! %d\n", err);
			exit(1);
		}
		clock_t end = clock();

		err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to retrieve kernel work group info! %d\n", err);
			exit(1);
		}

		global = DATA_SIZE;
		err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err)
		{
			printf("Error: Failed to execute kernel!\n");
			return EXIT_FAILURE;
		}

		clFinish(commands);
		clFlush(commands);

		printf("GB/s = %f\n",(float)(DATA_SIZE*2*(sizeof(float)))*(end - start)/CLOCKS_PER_SEC);
		printf("Time CPI CPU-GPU_CPU %fs\n",(float)(end - start)/CLOCKS_PER_SEC);

		correct = 0;
		for(int i = 0; i < DATA_SIZE; i++){
			if(results[i] == data[i] * data[i] && results[i] != 0)
				correct++;
		}	

		printf("Computed '%d/%d' correct values!\n", correct, DATA_SIZE);
		sleep(1);
	}

	// demapear mapear la memoria de la gpu en la cpu
	clEnqueueUnmapMemObject(commands, output, results, 0, NULL, NULL); 
	clEnqueueUnmapMemObject(commands, input, data, 0, NULL, NULL); 

    // Shutdown and cleanup

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}
