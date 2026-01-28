#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    // инициализация mpi окружения
    MPI_Init(&argc, &argv);
    
    // получаем общее количество процессов
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // получаем ранг текущего процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // размер массива данных
    const int N = 1000000;
    
    // массив для хранения всех данных (только на главном процессе)
    std::vector<double> data;
    
    // массив для хранения количества элементов для каждого процесса
    std::vector<int> sendcounts(world_size);
    
    // массив для хранения смещений (откуда начинать отправку для каждого процесса)
    std::vector<int> displs(world_size);
    
    // переменная для измерения времени начала
    double start_time = MPI_Wtime();
    
    // главный процесс создает и инициализирует данные
    if (rank == 0) {
        // инициализируем генератор случайных чисел
        srand(time(NULL));
        
        // выделяем память для всего массива
        data.resize(N);
        
        // заполняем массив случайными числами от 0 до 100
        for (int i = 0; i < N; i++) {
            data[i] = (double)(rand() % 100);
        }
        
        // вычисляем базовый размер подмассива для каждого процесса
        int base_size = N / world_size;
        
        // вычисляем остаток (элементы, которые не делятся нацело)
        int remainder = N % world_size;
        
        // распределяем элементы между процессами
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            // если есть остаток, первые процессы получают на 1 элемент больше
            sendcounts[i] = base_size + (i < remainder ? 1 : 0);
            
            // сохраняем смещение (откуда начинается подмассив для процесса i)
            displs[i] = offset;
            
            // обновляем смещение для следующего процесса
            offset += sendcounts[i];
        }
        
        std::cout << "Массив создан. Размер: " << N << std::endl;
        std::cout << "Количество процессов: " << world_size << std::endl;
    }
    
    // передаем информацию о размерах всем процессам
    MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // каждый процесс узнает размер своего подмассива
    int local_size = sendcounts[rank];
    
    // выделяем память для локального подмассива
    std::vector<double> local_data(local_size);
    
    // главный процесс распределяет данные между всеми процессами
    MPI_Scatterv(
        data.data(),           // буфер отправки (только на главном процессе)
        sendcounts.data(),     // массив с количеством элементов для каждого процесса
        displs.data(),         // массив со смещениями
        MPI_DOUBLE,            // тип отправляемых данных
        local_data.data(),     // буфер приема
        local_size,            // количество принимаемых элементов
        MPI_DOUBLE,            // тип принимаемых данных
        0,                     // ранг процесса-отправителя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // каждый процесс вычисляет локальную сумму своих элементов
    double local_sum = 0.0;
    for (int i = 0; i < local_size; i++) {
        local_sum += local_data[i];
    }
    
    // каждый процесс вычисляет локальную сумму квадратов своих элементов
    double local_sum_squares = 0.0;
    for (int i = 0; i < local_size; i++) {
        local_sum_squares += local_data[i] * local_data[i];
    }
    
    // переменные для хранения глобальных сумм (на главном процессе)
    double global_sum = 0.0;
    double global_sum_squares = 0.0;
    
    // собираем локальные суммы на главном процессе (операция сложения)
    MPI_Reduce(
        &local_sum,        // адрес отправляемого значения
        &global_sum,       // адрес для результата (только на главном)
        1,                 // количество элементов
        MPI_DOUBLE,        // тип данных
        MPI_SUM,           // операция (сложение)
        0,                 // ранг процесса-получателя
        MPI_COMM_WORLD     // коммуникатор
    );
    
    // собираем локальные суммы квадратов на главном процессе
    MPI_Reduce(
        &local_sum_squares,    // адрес отправляемого значения
        &global_sum_squares,   // адрес для результата
        1,                     // количество элементов
        MPI_DOUBLE,            // тип данных
        MPI_SUM,               // операция (сложение)
        0,                     // ранг процесса-получателя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // главный процесс вычисляет и выводит результаты
    if (rank == 0) {
        // вычисляем среднее значение
        double mean = global_sum / N;
        
        // вычисляем дисперсию: E[X^2] - (E[X])^2
        double variance = (global_sum_squares / N) - (mean * mean);
        
        // вычисляем стандартное отклонение (корень из дисперсии)
        double std_dev = sqrt(variance);
        
        // измеряем время окончания
        double end_time = MPI_Wtime();
        
        // выводим результаты
        std::cout << "\nРезультаты вычислений:" << std::endl;
        std::cout << "Среднее значение: " << mean << std::endl;
        std::cout << "Стандартное отклонение: " << std_dev << std::endl;
        std::cout << "Время выполнения: " << end_time - start_time << " секунд" << std::endl;
    }
    
    // завершаем работу mpi окружения
    MPI_Finalize();
    
    return 0;
}
