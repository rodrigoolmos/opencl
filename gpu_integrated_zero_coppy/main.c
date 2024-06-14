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

#define EPSILON 0.002
#define WIDTH 512
#define HEIGH 512


typedef struct
{
    float real;
    float imag;
} Complex;

////////////////////////////////////////////////////////////////////////////////

const char *KernelSource = ""
"__kernel void FFT(__global float2 *x, int N1, int N2)"
"{"
"    int fft_num = get_global_id(0);"
""
"    int i, j, m, len;"
"    float angle;"
"    float2 temp, wlen, w, u, v, next_w;"
""
"    j = 0;"
"    for (i = 0; i < N1; ++i)"
"    {"
"        if (i < j)"
"        {"
"            temp = x[fft_num * N1 + i];"
"            x[fft_num * N1 + i] = x[fft_num * N1 + j];"
"            x[fft_num * N1 + j] = temp;"
"        }"
"        for (m = N1 >> 1; m >= 1 && j >= m; m >>= 1)"
"        {"
"            j -= m;"
"        }"
"        j += m;"
"    }"
""
"    for (len = 2; len <= N1; len <<= 1)"
"    {"
"        angle = -2 * M_PI / len;"
"        wlen.x = cos(angle);"
"        wlen.y = sin(angle);"
"        for (i = 0; i < N1; i += len)"
"        {"
"            w.x = 1.0;"
"            w.y = 0.0;"
"            for (j = 0; j < len / 2; ++j)"
"            {"
"                u = x[fft_num * N1 + i + j];"
"                v.x = x[fft_num * N1 + i + j + len / 2].x * w.x - x[fft_num * N1 + i + j + len / 2].y * w.y;"
"                v.y = x[fft_num * N1 + i + j + len / 2].x * w.y + x[fft_num * N1 + i + j + len / 2].y * w.x;"
"                x[fft_num * N1 + i + j].x = u.x + v.x;"
"                x[fft_num * N1 + i + j].y = u.y + v.y;"
"                x[fft_num * N1 + i + j + len / 2].x = u.x - v.x;"
"                x[fft_num * N1 + i + j + len / 2].y = u.y - v.y;"
"                next_w.x = w.x * wlen.x - w.y * wlen.y;"
"                next_w.y = w.x * wlen.y + w.y * wlen.x;"
"                w = next_w;"
"            }"
"        }"
"    }"
"}"

;


////////////////////////////////////////////////////////////////////////////////

void FFT(Complex *x, int N1, int N2)
{
    int i, j, m, len, fft_num;
    float angle;
    Complex temp, wlen, w, u, v, next_w;

    for (fft_num = 0; fft_num < N2; fft_num++)
    {

        // Reordenamiento por bit reversal
        j = 0;
        for (i = 0; i < N1; ++i)
        {
            if (i < j)
            {
                temp = x[fft_num * N1 + i];
                x[fft_num * N1 + i] = x[fft_num * N1 + j];
                x[fft_num * N1 + j] = temp;
            }
            for (m = N1 >> 1; m >= 1 && j >= m; m >>= 1)
            {
                j -= m;
            }
            j += m;
        }

        // Algoritmo de la mariposa
        for (len = 2; len <= N1; len <<= 1)
        {
            angle = -2 * M_PI / len;
            wlen.real = cos(angle);
            wlen.imag = sin(angle);
            for (i = 0; i < N1; i += len)
            {
                w.real = 1.0;
                w.imag = 0.0;
                for (j = 0; j < len / 2; ++j)
                {
                    u = x[fft_num * N1 + i + j];
                    v.real = x[fft_num * N1 + i + j + len / 2].real * w.real - x[fft_num * N1 + i + j + len / 2].imag * w.imag;
                    v.imag = x[fft_num * N1 + i + j + len / 2].real * w.imag + x[fft_num * N1 + i + j + len / 2].imag * w.real;
                    x[fft_num * N1 + i + j].real = u.real + v.real;
                    x[fft_num * N1 + i + j].imag = u.imag + v.imag;
                    x[fft_num * N1 + i + j + len / 2].real = u.real - v.real;
                    x[fft_num * N1 + i + j + len / 2].imag = u.imag - v.imag;
                    next_w.real = w.real * wlen.real - w.imag * wlen.imag;
                    next_w.imag = w.real * wlen.imag + w.imag * wlen.real;
                    w = next_w;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int err; // error code returned from api calls

    ////////////////////////////////////////////////////////////////////////////
    //////////////////////     GPU stuff  //////////////////////////////////////

    size_t global; // global domain size for our calculation
    size_t local;  // local domain size for our calculation

    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel kernel;          // compute kernel

    cl_mem input; // device memory used for the input array

    cl_platform_id platforms; // platfor to work with
    ////////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////    CPU DATA   //////////////////////////////////
    float data[WIDTH*HEIGH*2] __attribute__((aligned(64))) = {0}; // necesario alinear el comienzo de la memoria en opencl formato
    float *data_ptr = data;
    Complex data_cpu[WIDTH*HEIGH];

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

    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
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

    kernel = clCreateKernel(program, "FFT", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Crear un array de entrada estando alojado en la cpu  CL_MEM_USE_HOST_PTR, data_ptr
    input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * 
                            WIDTH * HEIGH *2, data_ptr, NULL);
    if (!input)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }

    data_ptr = (float *)clEnqueueMapBuffer(commands, input, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) *
                         WIDTH * HEIGH * 2, 0, NULL, NULL, NULL);

    for (int i = 0; i < WIDTH * HEIGH  * 2; i++){
        data[i] = rand() / (float)RAND_MAX;
    }

    memcpy(data_cpu, data, sizeof(float) * WIDTH * HEIGH * 2);

    FFT(data_cpu, WIDTH, HEIGH);

    clock_t start = clock();

    err = 0;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    uint32_t width = 1;
    err |= clSetKernelArg(kernel, 1, sizeof(unsigned int), &width);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &width);

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

    global = HEIGH;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    clFinish(commands);
    clFlush(commands);

    int error = 0;
    for (int i = 0; i < WIDTH * HEIGH  * 2; i++){
        if(i % 2 == 0){
            if(data[i] != data_cpu[i].real){
                error = 1;
            }
        }else{
            if(data[i] != data_cpu[i].imag){
                error = 1;
            }
        }
    }

    // demapear mapear la memoria de la gpu en la cpu
    clEnqueueUnmapMemObject(commands, input, data, 0, NULL, NULL);

    // Shutdown and cleanup

    clReleaseMemObject(input);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    printf("STATUS %i\n", error);

    return 0;
}
