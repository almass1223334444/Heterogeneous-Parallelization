#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>

#define N 4
#define M 4
#define K 4

std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

int main() {
    // ---------------- МАТРИЦЫ ----------------
    std::vector<float> A(N * M, 1.0f);
    std::vector<float> B(M * K, 2.0f);
    std::vector<float> C(N * K);

    // ---------------- OPENCL ----------------
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, nullptr);

    // ---------------- БУФЕРЫ ----------------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N * M, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * M * K, B.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * N * K, nullptr, nullptr);

    // ---------------- ЯДРО ----------------
    std::string source = loadKernel("kernel_matrix_mul.cl");
    const char* src = source.c_str();
    size_t srcSize = source.size();

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcSize, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);
    clSetKernelArg(kernel, 4, sizeof(int), &M);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    // ---------------- ЗАПУСК ----------------
    size_t globalSize[2] = {N, K};
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * N * K, C.data(), 0, nullptr, nullptr);

    // ---------------- ВЫВОД ----------------
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++)
            std::cout << C[i * K + j] << " ";
        std::cout << "\n";
    }

    // ---------------- ОЧИСТКА ----------------
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
