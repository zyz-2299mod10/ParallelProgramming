__kernel void convolution(int filterWidth, __constant float *filter, int imageHeight, int imageWidth,
                          __global float *inputImage, __global float *outputImage)
{
    int halfFilterSize = filterWidth / 2;

    float sum = 0;
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    for (int k = -halfFilterSize; k <= halfFilterSize; k++) {
        for (int l = -halfFilterSize; l <= halfFilterSize; l++) {
            int neighborY = iy + k;
            int neighborX = ix + l;

            if (neighborY >= 0 && neighborY < imageHeight && neighborX >= 0 && neighborX < imageWidth) {
                int imageIndex = neighborY * imageWidth + neighborX;
                int filterIndex = (k + halfFilterSize) * filterWidth + (l + halfFilterSize);

                sum += inputImage[imageIndex] * filter[filterIndex];
            }
        }
    }

    int outputIndex = iy * imageWidth + ix;
    outputImage[outputIndex] = sum;
}