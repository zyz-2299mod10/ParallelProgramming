#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define GROUP_SIZE 5

__global__ void mandelKernel(int *deviceBuffer, float xMin, float yMin, float xStep, float yStep, int resolutionX, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize 
    float realPart = xMin + pixelX * xStep;
    float imagPart = yMin + pixelY * yStep;
    float zReal = realPart, zImag = imagPart;

    // Check if the point belongs to Mandelbrot set by |c| <= 0.25
    if (zReal * zReal + zImag * zImag <= 0.25f * 0.25f) {
        deviceBuffer[pixelY * resolutionX + pixelX] = maxIterations;
        return;
    }

    // Mandelbrot iterations
    int iterCount;
    for (iterCount = 0; iterCount < maxIterations; iterCount++) {
        if (zReal * zReal + zImag * zImag > 4.f) break;

        float tempReal = zReal * zReal - zImag * zImag;
        float tempImag = 2.f * zReal * zImag;
        zReal = realPart + tempReal;
        zImag = imagPart + tempImag;
    }

    // Store 
    deviceBuffer[pixelY * resolutionX + pixelX] = iterCount;
}

// Host function: Allocates memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    float xStep = (upperX - lowerX) / resX;
    float yStep = (upperY - lowerY) / resY;

    // Total number of pixels
    int totalPixels = resX * resY;

    // Map host memory to device memory
    int *deviceBuffer;
    cudaHostRegister(img, totalPixels * sizeof(int), cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&deviceBuffer, img, 0);

    // GPU kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((resX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (resY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mandelKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceBuffer, lowerX, lowerY, xStep, yStep, resX, maxIterations);

    // Wait 
    cudaDeviceSynchronize();

    // Unregister the mapped memory
    cudaHostUnregister(img);

    // Copy data back from device to host memory
    cudaMemcpy(img, deviceBuffer, totalPixels * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(deviceBuffer);
}
