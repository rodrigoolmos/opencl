#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

struct gpu_square
{
	// recusos de la GPU siempre lo mismo
	cl_device_id device_id;             
    cl_context context;                 
    cl_command_queue commands;          
    cl_program program;                 
    cl_kernel kernel;

	// buffers entrada salida en funcion de las necesidades 
    cl_mem input;
	uint32_t size_input;                 
    cl_mem output; 
	uint32_t size_output;                       
};


struct gpu_adition
{
	// recusos de la GPU siempre lo mismo
	cl_device_id device_id;             
    cl_context context;                 
    cl_command_queue commands;          
    cl_program program;                 
    cl_kernel kernel;

	// buffers entrada salida en funcion de las necesidades 
	cl_mem input2;
	uint32_t size_input2;                    
    cl_mem output; 
	uint32_t size_output;                       
};


struct gpu_square gpu_square;
struct gpu_adition gpu_adition;

////////////////////////////////////////////////////////////////////////////////

// Use a static data size for simplicity
//
#define DATA_SIZE (1024)

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char *kernel_square = "\n" \
"__kernel void square(                                                  \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"	if(i==0)                                                            \n" \
"		printf(\"%f \\n\",output[0]);									\n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the adition of a 2 input arrays
//
const char *kernel_adition = "\n" \
"__kernel void adition(                                                 \n" \
"   __global float* input1,                                             \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input1[i] + input2[i];                              \n" \
"	if(i==0)                                                            \n" \
"		printf(\"%f \\n\",input1[0]);									\n" \
"}                                                                      \n" \
"\n";

////////////////////////////////////////////////////////////////////////////////

uint32_t init_gpu_square (){
	
	cl_platform_id platforms;			// platfor to work with
	cl_int err; 
	    // Connect to a compute device
    //
    err = clGetPlatformIDs(1, &platforms, NULL);
    err = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &(gpu_square.device_id), NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    gpu_square.context = clCreateContext(0, 1, &(gpu_square.device_id), NULL, NULL, &err);
    if (!gpu_square.context){
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    gpu_square.commands = clCreateCommandQueue(gpu_square.context, gpu_square.device_id, 0, &err);
    if (!gpu_square.commands){
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    gpu_square.program = clCreateProgramWithSource(gpu_square.context, 1, (const char **) & kernel_square, NULL, &err);
    if (!gpu_square.program){
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(gpu_square.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(gpu_square.program, gpu_square.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    } 

    // Create the compute kernel in the program we wish to run
    //
    gpu_square.kernel = clCreateKernel(gpu_square.program, "square", &err);
    if (!gpu_square.kernel || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

	// Create the input and output arrays in device memory for our calculation
    gpu_square.input  = clCreateBuffer(gpu_square.context,  CL_MEM_READ_ONLY, sizeof(float) * gpu_square.size_input,  NULL, NULL);
    gpu_square.output = clCreateBuffer(gpu_square.context, CL_MEM_READ_WRITE, sizeof(float) * gpu_square.size_output, NULL, NULL);
    if (!gpu_square.input || !gpu_square.output){
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

	//---------------------------------------------------------------------------------------------------------------------
	//  										ARGUMENTOS UNA VEZ AQUI
    //
    // 	err = clEnqueueWriteBuffer(gpu_adition.commands, gpu_adition.data_gpu, CL_TRUE, 0, sizeof(float) * gpu_adition.size_input, data_cpu, 0, NULL, NULL);
    // 	if (err != CL_SUCCESS){
    // 	    printf("Error: Failed to write to source array!\n");
    // 	    exit(1);
    // 	}
    // 	err  = clSetKernelArg(gpu_adition.kernel, position, sizeof(cl_mem), &(gpu_adition.input));
	//---------------------------------------------------------------------------------------------------------------------

	err  = clSetKernelArg(gpu_square.kernel, 1, sizeof(cl_mem), &(gpu_square.output));
	if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

	return err;
}

uint32_t execute_gpu_square (void* gpu_data_in, void* gpu_data_out){

	cl_int err;
	size_t local;  
	size_t global;    
 
	//---------------------------------------------------------------------------------------------------------------------
	//  										ARGUMENTOS MAS DE UNA VEZ AQUI
    //
    err = clEnqueueWriteBuffer(gpu_square.commands, gpu_square.input, CL_TRUE, 0, sizeof(float) * gpu_square.size_input, gpu_data_in, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
    err  = clSetKernelArg(gpu_square.kernel, 0, sizeof(cl_mem), &(gpu_square.input));
    err |= clSetKernelArg(gpu_square.kernel, 2, sizeof(unsigned int), &(gpu_square.size_input));
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

	// 
    //----------------------------------------------------------------------------------------------------------------------

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(gpu_square.kernel, gpu_square.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = gpu_square.size_input;
    err = clEnqueueNDRangeKernel(gpu_square.commands, gpu_square.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }



	//--------------------------------------------------------------------------------------------------//
    //-- solo si necesito sacarlos a la cpu si no pasar el puntero gpu_square.output a el siguiente kernel  --//
	//--------------------------------------------------------------------------------------------------//
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(gpu_square.commands, gpu_square.output, CL_TRUE, 0, sizeof(float) * gpu_square.size_output, gpu_data_out, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

	err = clFlush(gpu_square.commands);
    err = clFinish(gpu_square.commands);

	return err;
}

uint32_t release_gpu_square (){

	cl_int err;
    // Shutdown and cleanup
    //
    err = clReleaseMemObject(gpu_square.input);
    err = clReleaseMemObject(gpu_square.output);
    err = clReleaseProgram(gpu_square.program);
    err = clReleaseKernel(gpu_square.kernel);
    err = clReleaseCommandQueue(gpu_square.commands);
    err = clReleaseContext(gpu_square.context);

	return err;
}


uint32_t init_gpu_adition (){
	
	cl_platform_id platforms;			// platfor to work with
	cl_int err; 
	    // Connect to a compute device
    //
    err = clGetPlatformIDs(1, &platforms, NULL);
    err = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &(gpu_adition.device_id), NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    gpu_adition.context = clCreateContext(0, 1, &(gpu_adition.device_id), NULL, NULL, &err);
    if (!gpu_adition.context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    gpu_adition.commands = clCreateCommandQueue(gpu_adition.context, gpu_adition.device_id, 0, &err);
    if (!gpu_adition.commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    gpu_adition.program = clCreateProgramWithSource(gpu_adition.context, 1, (const char **) & kernel_adition, NULL, &err);
    if (!gpu_adition.program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(gpu_adition.program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(gpu_adition.program, gpu_adition.device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    } 

    // Create the compute kernel in the program we wish to run
    //
    gpu_adition.kernel = clCreateKernel(gpu_adition.program, "adition", &err);
    if (!gpu_adition.kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

	// Create the input and output arrays in device memory for our calculation
    // argumentos que solo se pasen una vez aqui
	gpu_adition.input2 = clCreateBuffer(gpu_adition.context,  CL_MEM_READ_ONLY,  sizeof(float) * gpu_adition.size_input2, NULL, NULL);
    gpu_adition.output = clCreateBuffer(gpu_adition.context,  CL_MEM_WRITE_ONLY, sizeof(float) * gpu_adition.size_output, NULL, NULL);
    if ( !gpu_adition.input2 || !gpu_adition.output)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }   

	//---------------------------------------------------------------------------------------------------------------------
	//  										ARGUMENTOS UNA VEZ AQUI
    //
    // 	err = clEnqueueWriteBuffer(gpu_adition.commands, gpu_adition.data_gpu, CL_TRUE, 0, sizeof(float) * gpu_adition.size_input, data_cpu, 0, NULL, NULL);
    // 	if (err != CL_SUCCESS){
    // 	    printf("Error: Failed to write to source array!\n");
    // 	    exit(1);
    // 	}
    // 	err  = clSetKernelArg(gpu_adition.kernel, position, sizeof(cl_mem), &(gpu_adition.input));
	//---------------------------------------------------------------------------------------------------------------------

	return err;
}

uint32_t execute_gpu_adition (const void* gpu_data_in2, void* gpu_data_out){

	cl_int err;
	size_t local;  
	size_t global;    
 
	//---------------------------------------------------------------------------------------------------------------------
	//  										ARGUMENTOS MAS DE UNA VEZ AQUI
	//
    err = clEnqueueWriteBuffer(gpu_adition.commands, gpu_adition.input2, CL_TRUE, 0, sizeof(float) * gpu_adition.size_input2, gpu_data_in2, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }
	
    err  = clSetKernelArg(gpu_adition.kernel, 0, sizeof(cl_mem), &(gpu_square.output));
	err |= clSetKernelArg(gpu_adition.kernel, 1, sizeof(cl_mem), &(gpu_adition.input2));
    err |= clSetKernelArg(gpu_adition.kernel, 2, sizeof(cl_mem), &(gpu_adition.output));
    err |= clSetKernelArg(gpu_adition.kernel, 3, sizeof(unsigned int), &(gpu_adition.size_input2));
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
	//
	//---------------------------------------------------------------------------------------------------------------------

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(gpu_adition.kernel, gpu_adition.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = gpu_adition.size_input2;
    err = clEnqueueNDRangeKernel(gpu_adition.commands, gpu_adition.kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    //
	err = clFlush(gpu_adition.commands);
    err = clFinish(gpu_adition.commands);

    // Read back the results from the device to verify the output
	//--------------------------------------------------------------------------------------------------//
    //-- solo si necesito sacarlos a la cpu si no pasar el puntero gpu_adition.output a el siguiente kernel  --//
	//--------------------------------------------------------------------------------------------------//
    err = clEnqueueReadBuffer(gpu_adition.commands, gpu_adition.output, CL_TRUE, 0, sizeof(float) * gpu_adition.size_output, gpu_data_out, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

	return err;
}

uint32_t release_gpu_adition (){

	cl_int err;
    // Shutdown and cleanup
    //
    err = clReleaseMemObject(gpu_adition.input2);
    err = clReleaseMemObject(gpu_adition.output);
    err = clReleaseProgram(gpu_adition.program);
    err = clReleaseKernel(gpu_adition.kernel);
    err = clReleaseCommandQueue(gpu_adition.commands);
    err = clReleaseContext(gpu_adition.context);

	return err;
}


int main(int argc, char** argv)
{  

	gpu_adition.size_input2 = 1024;
	gpu_adition.size_output = 1024;
	gpu_square.size_input = 1024;
	gpu_square.size_output = 1024;

    float data[DATA_SIZE];
	float data2[DATA_SIZE];                
    float results[DATA_SIZE];           
    unsigned int correct;               
	
    size_t global;                      
    size_t local;                       

	
    // Fill our data set with random float values
    //
    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++){
        data[i] = rand() / (float)RAND_MAX;
		data2[i] = rand() / (float)RAND_MAX;
	}

	init_gpu_square();
	init_gpu_adition();

	execute_gpu_square(data,results);

    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i])
            correct++;
    }

	printf("Computed '%d/%d' correct values!\n", correct, count);


	execute_gpu_adition(data2,results);
    
    // Validate our results
    //
    correct = 0;
    for(i = 0; i < count; i++)
    {
        if(results[i] == data[i] * data[i] + data2[i])
            correct++;
    }
    
    // Print a brief summary detailing the results
    //
    printf("Computed '%d/%d' correct values!\n", correct, count);
    
	release_gpu_square();
	release_gpu_adition();

    return 0;
}
