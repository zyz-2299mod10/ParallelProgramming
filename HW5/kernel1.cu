#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *outputData, float xMin, float yMin, float xStep, float yStep, int width, int maxIter) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    float realPart = xMin + pixelX * xStep;
    float imagPart = yMin + pixelY * yStep;
    float zReal = realPart, zImag = imagPart;

    // Check if the point lies within the |c| <= 0.25 
    if (zReal * zReal + zImag * zImag <= 0.25f * 0.25f) {
        outputData[pixelY * width + pixelX] = maxIter;
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
    outputData[pixelY * width + pixelX] = iteration;
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations) {
    float xStep = (upperX - lowerX) / resX;
    float yStep = (upperY - lowerY) / resY;

    // Allocate memory 
    int totalPixels = resX * resY;
    int *hostOutput = (int*)malloc(totalPixels * sizeof(int));
    int *deviceOutput;
    cudaMalloc(&deviceOutput, totalPixels * sizeof(int));

    // GPU kernel
    dim3 threadsPerBlock(20, 20);
    dim3 blocksPerGrid((resX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (resY + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mandelKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput, lowerX, lowerY, xStep, yStep, resX, maxIterations);

    // wait
    cudaDeviceSynchronize();

    // Copy results from device to host memory
    cudaMemcpy(hostOutput, deviceOutput, totalPixels * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, hostOutput, totalPixels * sizeof(int));

    // Free memory
    cudaFree(deviceOutput);
    free(hostOutput);
}
