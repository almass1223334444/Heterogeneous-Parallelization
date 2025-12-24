__device__ void quickSort(int* arr, int left, int right) {
    int i = left, j = right;
    int pivot = arr[(left + right) / 2];

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++; j--;
        }
    }
}

__global__ void quickSortKernel(int* data, int chunk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * chunk;
    int end = start + chunk - 1;

    quickSort(data, start, end);
}
