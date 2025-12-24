__device__ void heapify(int* arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
    }
}

__global__ void heapSortKernel(int* arr, int n) {
    int i = threadIdx.x;
    if (i < n / 2)
        heapify(arr, n, i);
}
