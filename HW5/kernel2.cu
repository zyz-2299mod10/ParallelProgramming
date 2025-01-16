#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *deviceData, float xMin, float yMin, float xStep, float yStep, size_t rowPitch, int maxIter) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    float realPart = xMin + pixelX * xStep;
    float imagPart = yMin + pixelY * yStep;
    float zReal = realPart, zImag = imagPart;

    // Get a pointer to the pixel this thread will process
    int* pixelPtr = (int*)((char*)deviceData + pixelY * rowPitch) + pixelX;

    // Check if the point lies within the |c| <= 0.25 
    if (zReal * zReal + zImag * zImag <= 0.25f * 0.25f) {
        *pixelPtr = maxIter;
        return;
    }

    // Mandelbrot iteration
    int iteration;
    for (iteration = 0; iteration < maxIter; iteration++) {
        if (zReal * zReal + zImag * zImag > 4.f) break;

        float tempReal = zReal * zReal - zImag * zImag;
        float tempImag = 2.f * zReal * zImag;
        zReal = realPart + tempReal;
        zImag = imagPart + tempImag;
    }

    // Store 
    *pixelPtr = iteration;
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    float xStep = (upperX - lowerX) / resX;
    float yStep = (upperY - lowerY) / resY;

    // Allocate pinned host memory
    int totalPixels = resX * resY;
    int *hostOutput;
    cudaHostAlloc((void**)&hostOutput, totalPixels * sizeof(int), cudaHostAllocMapped);

    // Allocate pitched memory
    int *deviceOutput;
    size_t deviceRowPitch;
    cudaMallocPitch(&deviceOutput, &deviceRowPitch, resX * sizeof(int), resY);
    
    // GPU kernel
    dim3 threadsPerBlock(20, 20);
    dim3 blocksPerGrid((resX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (resY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mandelKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput, lowerX, lowerY, xStep, yStep, deviceRowPitch, maxIterations);

    // Wait 
    cudaDeviceSynchronize();

    // Copy results from device to host memory
    cudaMemcpy2D(hostOutput, resX * sizeof(int), deviceOutput, deviceRowPitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, hostOutput, totalPixels * sizeof(int));

    // Free memory
    cudaFree(deviceOutput);
    cudaFreeHost(hostOutput);
}
