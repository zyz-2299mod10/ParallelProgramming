#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    size_t imageSize = imageHeight * imageWidth * sizeof(float);

    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");

    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize, NULL, &status);
    CHECK(status, "clCreateBuffer (inputBuffer)");

    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, &status);
    CHECK(status, "clCreateBuffer (outputBuffer)");

    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, &status);
    CHECK(status, "clCreateBuffer (filterBuffer)");

    status = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, imageSize, inputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer (inputBuffer)");

    status = clEnqueueWriteBuffer(queue, filterBuffer, CL_TRUE, 0, filterSize, filter, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer (filterBuffer)");

    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // kernel parameter
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), &filterWidth);
    CHECK(status, "clSetKernelArg (filterWidth)");

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    CHECK(status, "clSetKernelArg (filterBuffer)");

    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &imageHeight);
    CHECK(status, "clSetKernelArg (imageHeight)");

    status = clSetKernelArg(kernel, 3, sizeof(cl_int), &imageWidth);
    CHECK(status, "clSetKernelArg (imageWidth)");

    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &inputBuffer);
    CHECK(status, "clSetKernelArg (inputBuffer)");

    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &outpupptBuffer);
    CHECK(status, "clSetKernelArg (outputBuffer)");

    size_t localWorkSize[2] = {20, 20};
    size_t globalWorkSize[2] = {imageWidth, imageHeight};

    // exec kernel
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");

    status = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");

    clReleaseKernel(kernel);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseCommandQueue(queue);
}