#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define GROUP_SIZE 5

__global__ void mandelKernel(int *deviceBuffer, float xMin, float yMin, float xStep, float yStep, size_t rowPitch, int maxIter) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    int baseX = (blockIdx.x * blockDim.x + threadIdx.x) * GROUP_SIZE;
    int baseY = (blockIdx.y * blockDim.y + threadIdx.y) * GROUP_SIZE;

    // Process a group of pixels within this thread
    for (int offsetY = 0; offsetY < GROUP_SIZE; offsetY++) {
        for (int offsetX = 0; offsetX < GROUP_SIZE; offsetX++) {
            // Calculate the local pixel coordinates
            int pixelX = baseX + offsetX;
            int pixelY = baseY + offsetY;

            float realPart = xMin + pixelX * xStep;
            float imagPart = yMin + pixelY * yStep;
            float zReal = realPart, zImag = imagPart;

            // Get a pointer to the pixel data
            int* pixelPtr = (int*)((char*)deviceBuffer + pixelY * rowPitch) + pixelX;

            // Check if the point lies within the |c| <= 0.25  
            if (zReal * zReal + zImag * zImag <= 0.25f * 0.25f) {
                *pixelPtr = maxIter;
                continue;
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
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {

    float xStep = (upperX - lowerX) / resX;
    float yStep = (upperY - lowerY) / resY;

    // Allocate pinned host memory
    int totalPixels = resX * resY;
    int *hostBuffer;
    cudaHostAlloc((void**)&hostBuffer, totalPixels * sizeof(int), cudaHostAllocMapped);

    // Allocate pitched memory on the device
    int *deviceBuffer;
    size_t deviceRowPitch;
    cudaMallocPitch(&deviceBuffer, &deviceRowPitch, resX * sizeof(int), resY);

    // GPU kernel
    dim3 threadsPerBlock(20, 20);
    dim3 blocksPerGrid((resX + threadsPerBlock.x * GROUP_SIZE - 1) / (threadsPerBlock.x * GROUP_SIZE),
                       (resY + threadsPerBlock.y * GROUP_SIZE - 1) / (threadsPerBlock.y * GROUP_SIZE));
    mandelKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceBuffer, lowerX, lowerY, xStep, yStep, deviceRowPitch, maxIterations);

    // Wait 
    cudaDeviceSynchronize();

    // Copy results from device to host memory
    cudaMemcpy2D(hostBuffer, resX * sizeof(int), deviceBuffer, deviceRowPitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, hostBuffer, totalPixels * sizeof(int));

    // Free memory
    cudaFree(deviceBuffer);
    cudaFreeHost(hostBuffer);
}
