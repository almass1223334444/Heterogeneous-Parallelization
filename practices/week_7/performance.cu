#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

float cpu_sum(const std::vector<float>& data) {
    float sum = 0.0f;
    for (float x : data)
        sum += x;
    return sum;
}

int main() {

    const int N = 1'000'000;
    std::vector<float> data(N, 1.0f);

    // CPU измерение
    auto start = std::chrono::high_resolution_clock::now();
    float cpuResult = cpu_sum(data);
    auto end = std::chrono::high_resolution_clock::now();

    double cpuTime = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "CPU sum = " << cpuResult << std::endl;
    std::cout << "CPU time = " << cpuTime << " ms" << std::endl;

    return 0;
}
