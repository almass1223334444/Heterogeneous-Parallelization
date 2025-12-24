#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

__global__ void mergeSortKernel(int* data, int size, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * step * 2;

    if (start + step < size) {
        int mid = start + step;
        int end = min(start + step * 2, size);

        int i = start, j = mid;
        int k = 0;
        extern __shared__ int temp[];

        while (i < mid && j < end)
            temp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];

        while (i < mid) temp[k++] = data[i++];
        while (j < end) temp[k++] = data[j++];

        for (int n = 0; n < k; n++)
            data[start + n] = temp[n];
    }
}

void mergeSortCUDA(int* d_data, int size) {
    int threads = 256;
    int blocks;

    for (int step = 1; step < size; step *= 2) {
        blocks = (size / (step * 2)) + 1;
        mergeSortKernel<<<blocks, threads, step * 2 * sizeof(int)>>>(d_data, size, step);
        cudaDeviceSynchronize();
    }
}
