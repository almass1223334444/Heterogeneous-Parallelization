#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

// функция для тестирования масштабируемости задачи 1
void benchmark_task1(int rank, int world_size) {
    // размер массива данных
    const int N = 10000000; // увеличен для лучшего замера
    
    // массив для хранения всех данных (только на главном процессе)
    std::vector<double> data;
    
    // массив для хранения количества элементов для каждого процесса
    std::vector<int> sendcounts(world_size);
    
    // массив для хранения смещений
    std::vector<int> displs(world_size);
    
    // переменная для измерения времени начала
    double start_time = MPI_Wtime();
    
    // главный процесс создает данные
    if (rank == 0) {
        // инициализируем генератор случайных чисел
        srand(time(NULL));
        
        // выделяем память для всего массива
        data.resize(N);
        
        // заполняем массив случайными числами
        for (int i = 0; i < N; i++) {
            data[i] = (double)(rand() % 100);
        }
        
        // вычисляем базовый размер подмассива для каждого процесса
        int base_size = N / world_size;
        
        // вычисляем остаток
        int remainder = N % world_size;
        
        // распределяем элементы между процессами
        int offset = 0;
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = base_size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // передаем информацию о размерах всем процессам
    MPI_Bcast(sendcounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // каждый процесс узнает размер своего подмассива
    int local_size = sendcounts[rank];
    
    // выделяем память для локального подмассива
    std::vector<double> local_data(local_size);
    
    // главный процесс распределяет данные
    MPI_Scatterv(data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_data.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // каждый процесс вычисляет локальную сумму
    double local_sum = 0.0;
    for (int i = 0; i < local_size; i++) {
        local_sum += local_data[i];
    }
    
    // каждый процесс вычисляет локальную сумму квадратов
    double local_sum_squares = 0.0;
    for (int i = 0; i < local_size; i++) {
        local_sum_squares += local_data[i] * local_data[i];
    }
    
    // переменные для хранения глобальных сумм
    double global_sum = 0.0;
    double global_sum_squares = 0.0;
    
    // собираем локальные суммы на главном процессе
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_squares, &global_sum_squares, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // главный процесс вычисляет результаты
    if (rank == 0) {
        // вычисляем среднее значение
        double mean = global_sum / N;
        
        // вычисляем дисперсию
        double variance = (global_sum_squares / N) - (mean * mean);
        
        // вычисляем стандартное отклонение
        double std_dev = sqrt(variance);
        
        // измеряем время окончания
        double end_time = MPI_Wtime();
        
        // выводим результаты бенчмарка
        std::cout << "Задание 1 - Размер данных: " << N << std::endl;
        std::cout << "Процессов: " << world_size << std::endl;
        std::cout << "Время выполнения: " << std::fixed << std::setprecision(6) 
                  << end_time - start_time << " сек" << std::endl;
        std::cout << "Среднее: " << mean << ", Ст.откл: " << std_dev << std::endl;
        std::cout << std::endl;
    }
}

// функция для тестирования масштабируемости задачи 3
void benchmark_task3(int rank, int world_size) {
    // размер графа (количество вершин)
    const int N = 512; // степень двойки для равномерного распределения
    
    // проверяем делимость
    if (N % world_size != 0) {
        if (rank == 0) {
            std::cout << "Ошибка: размер графа должен делиться на количество процессов" << std::endl;
        }
        return;
    }
    
    // константа для представления бесконечности
    const double INF = 1e9;
    
    // вычисляем количество строк для каждого процесса
    int rows_per_process = N / world_size;
    
    // матрица смежности графа
    std::vector<double> graph;
    
    // измеряем время начала
    double start_time = MPI_Wtime();
    
    // главный процесс создает граф
    if (rank == 0) {
        // выделяем память для матрицы смежности
        graph.resize(N * N);
        
        // инициализируем генератор случайных чисел
        srand(42);
        
        // заполняем матрицу смежности
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) {
                    graph[i * N + j] = 0;
                } else {
                    // с вероятностью 30% создаем ребро (разреженный граф)
                    if (rand() % 100 < 30) {
                        graph[i * N + j] = (rand() % 20) + 1;
                    } else {
                        graph[i * N + j] = INF;
                    }
                }
            }
        }
    }
    
    // каждый процесс выделяет память для своей части матрицы
    std::vector<double> local_graph(rows_per_process * N);
    
    // распределяем строки матрицы между процессами
    MPI_Scatter(graph.data(), rows_per_process * N, MPI_DOUBLE,
                local_graph.data(), rows_per_process * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // буфер для хранения k-й строки
    std::vector<double> k_row(N);
    
    // основной цикл алгоритма флойда-уоршелла
    for (int k = 0; k < N; k++) {
        // определяем процесс владелец k-й строки
        int k_process = k / rows_per_process;
        
        // если текущий процесс владеет k-й строкой
        if (rank == k_process) {
            // локальный индекс k-й строки
            int local_k = k % rows_per_process;
            
            // копируем k-ю строку в буфер
            for (int j = 0; j < N; j++) {
                k_row[j] = local_graph[local_k * N + j];
            }
        }
        
        // рассылаем k-ю строку всем процессам
        MPI_Bcast(k_row.data(), N, MPI_DOUBLE, k_process, MPI_COMM_WORLD);
        
        // каждый процесс обновляет свои строки
        for (int i = 0; i < rows_per_process; i++) {
            for (int j = 0; j < N; j++) {
                // текущее расстояние от i до j
                double current_dist = local_graph[i * N + j];
                
                // расстояние через промежуточную вершину k
                double new_dist = local_graph[i * N + k] + k_row[j];
                
                // если новый путь короче, обновляем
                if (new_dist < current_dist) {
                    local_graph[i * N + j] = new_dist;
                }
            }
        }
    }
    
    // собираем результаты на главном процессе
    MPI_Gather(local_graph.data(), rows_per_process * N, MPI_DOUBLE,
               graph.data(), rows_per_process * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    
    // главный процесс выводит результаты
    if (rank == 0) {
        // измеряем время окончания
        double end_time = MPI_Wtime();
        
        std::cout << "Задание 3 - Размер графа: " << N << "x" << N << std::endl;
        std::cout << "Процессов: " << world_size << std::endl;
        std::cout << "Время выполнения: " << std::fixed << std::setprecision(6) 
                  << end_time - start_time << " сек" << std::endl;
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    // инициализация mpi окружения
    MPI_Init(&argc, &argv);
    
    // получаем общее количество процессов
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // получаем ранг текущего процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // выводим заголовок
    if (rank == 0) {
        std::cout << "=====================================" << std::endl;
        std::cout << "БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ MPI" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << std::endl;
    }
    
    // тестируем задание 1
    if (rank == 0) {
        std::cout << "--- Тест 1: Вычисление статистики ---" << std::endl;
    }
    benchmark_task1(rank, world_size);
    
    // синхронизируем все процессы перед следующим тестом
    MPI_Barrier(MPI_COMM_WORLD);
    
    // тестируем задание 3
    if (rank == 0) {
        std::cout << "--- Тест 2: Алгоритм Флойда-Уоршелла ---" << std::endl;
    }
    benchmark_task3(rank, world_size);
    
    // синхронизируем все процессы
    MPI_Barrier(MPI_COMM_WORLD);
    
    // выводим рекомендации
    if (rank == 0) {
        std::cout << "=====================================" << std::endl;
        std::cout << "РЕКОМЕНДАЦИИ:" << std::endl;
        std::cout << "1. Запустите с разным количеством процессов (2, 4, 8)" << std::endl;
        std::cout << "2. Сравните время выполнения" << std::endl;
        std::cout << "3. Постройте графики зависимости времени от количества процессов" << std::endl;
        std::cout << "4. Вычислите коэффициент ускорения: T1 / Tn" << std::endl;
        std::cout << "5. Оцените эффективность: Ускорение / Количество процессов" << std::endl;
        std::cout << "=====================================" << std::endl;
    }
    
    // завершаем работу mpi окружения
    MPI_Finalize();
    
    return 0;
}
