#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "hostFE.h"
}
#include "helper.h"

__constant__ float const_filter[500];

__global__ void convolution(int filterWidth, int imageHeight, int imageWidth,
                            float *inputImage, float *outputImage) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int halfFilterSize = filterWidth / 2;
    float sum = 0;

    for (int k = -halfFilterSize; k <= halfFilterSize; k++) {
        for (int l = -halfFilterSize; l <= halfFilterSize; l++) {
            int neighborY = iy + k;
            int neighborX = ix + l;

            // Check boundary conditions to avoid accessing invalid memory
            if (neighborY >= 0 && neighborY < imageHeight &&
                neighborX >= 0 && neighborX < imageWidth) {
                int imageIndex = neighborY * imageWidth + neighborX;
                int filterIndex = (k + halfFilterSize) * filterWidth + (l + halfFilterSize);

                sum += inputImage[imageIndex] * const_filter[filterIndex];
            }
        }
    }

    // Store 
    if (iy < imageHeight && ix < imageWidth) { // Ensure within valid range
        int outputIndex = iy * imageWidth + ix;
        outputImage[outputIndex] = sum;
    }
}

extern "C"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
                
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    size_t imageSize = imageHeight * imageWidth * sizeof(float);

    // Allocate device memory 
    float *deviceInputImage, *deviceOutputImage;
    cudaMalloc(&deviceInputImage, imageSize);
    cudaMalloc(&deviceOutputImage, imageSize);

    cudaMemcpyToSymbol(const_filter, filter, filterSize, 0, cudaMemcpyHostToDevice);

    cudaMemcpy(deviceInputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    // CUDA kernel 
    dim3 threadsPerBlock(15, 15);
    dim3 numBlocks((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolution<<<numBlocks, threadsPerBlock>>>(filterWidth, imageHeight, imageWidth,
                                                deviceInputImage, deviceOutputImage);

    // Wait 
    cudaDeviceSynchronize();

    // Copy the result 
    cudaMemcpy(outputImage, deviceOutputImage, imageSize, cudaMemcpyDeviceToHost);

    // Free 
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
}
