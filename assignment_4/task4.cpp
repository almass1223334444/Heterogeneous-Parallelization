#include <mpi.h>           // MPI
#include <iostream>        // Ввод/вывод
#include <vector>          // std::vector

#define N 1000000          // Размер массива

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);                // Инициализация MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Количество процессов

    int chunk = N / size;                  // Размер части массива на процесс

    std::vector<float> data;
    if (rank == 0) {
        data.resize(N);                    // Только процесс 0 создаёт массив
        for (int i = 0; i < N; i++) data[i] = 1.0f;
    }

    std::vector<float> local(chunk);       // Локальный кусок массива

    MPI_Scatter(data.data(), chunk, MPI_FLOAT,   // Рассылка данных
                local.data(), chunk, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    float local_sum = 0;
    for (int i = 0; i < chunk; i++)           // Локальная обработка
        local_sum += local[i];

    float global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // Сбор результатов

    if (rank == 0) {
        std::cout << "Global sum = " << global_sum << std::endl;
    }

    MPI_Finalize();                          // Завершение MPI
    return 0;
}
