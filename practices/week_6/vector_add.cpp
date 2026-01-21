#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>

#define N 1000000

std::string loadKernel(const char* filename) {
    std::ifstream file(filename);
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

int main() {
    // ---------------- ДАННЫЕ ----------------
    std::vector<float> A(N, 1.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);

    // ---------------- OPENCL ----------------
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    cl_command_queue queue =
        clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, nullptr);

    // ---------------- БУФЕРЫ ----------------
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N, B.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * N, nullptr, nullptr);

    // ---------------- ЯДРО ----------------
    std::string source = loadKernel("kernel_vector_add.cl");
    const char* src = source.c_str();
    size_t srcSize = source.size();

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcSize, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    // ---------------- ЗАПУСК ----------------
    size_t globalSize = N;

    auto start = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    auto end = std::chrono::high_resolution_clock::now();

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * N, C.data(), 0, nullptr, nullptr);

    std::chrono::duration<double> time = end - start;
    std::cout << "GPU vector add time: " << time.count() << " sec\n";

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
