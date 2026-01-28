#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <limits>

// константа для представления бесконечности
const double INF = std::numeric_limits<double>::infinity();

int main(int argc, char** argv) {
    // инициализация mpi окружения
    MPI_Init(&argc, &argv);
    
    // получаем общее количество процессов
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // получаем ранг текущего процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // размер графа (количество вершин)
    const int N = 8;
    
    // проверяем, что размер графа делится на количество процессов
    if (N % world_size != 0) {
        if (rank == 0) {
            std::cout << "Ошибка: размер графа должен делиться на количество процессов" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // вычисляем количество строк для каждого процесса
    int rows_per_process = N / world_size;
    
    // матрица смежности графа (только на главном процессе)
    std::vector<double> graph;
    
    // измеряем время начала
    double start_time = MPI_Wtime();
    
    // главный процесс создает граф
    if (rank == 0) {
        // выделяем память для матрицы смежности
        graph.resize(N * N);
        
        // инициализируем генератор случайных чисел
        srand(42); // фиксированный seed для воспроизводимости
        
        // заполняем матрицу смежности
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) {
                    // расстояние от вершины до себя равно 0
                    graph[i * N + j] = 0;
                } else {
                    // с вероятностью 70% создаем ребро
                    if (rand() % 100 < 70) {
                        // случайный вес ребра от 1 до 20
                        graph[i * N + j] = (rand() % 20) + 1;
                    } else {
                        // нет ребра (бесконечность)
                        graph[i * N + j] = INF;
                    }
                }
            }
        }
        
        std::cout << "Граф создан. Количество вершин: " << N << std::endl;
        std::cout << "Количество процессов: " << world_size << std::endl;
        
        std::cout << "\nИсходная матрица смежности:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (graph[i * N + j] == INF) {
                    std::cout << std::setw(8) << "INF";
                } else {
                    std::cout << std::setw(8) << std::fixed << std::setprecision(1) << graph[i * N + j];
                }
            }
            std::cout << std::endl;
        }
    }
    
    // каждый процесс выделяет память для своей части матрицы
    std::vector<double> local_graph(rows_per_process * N);
    
    // распределяем строки матрицы между процессами
    MPI_Scatter(
        graph.data(),          // буфер отправки
        rows_per_process * N,  // количество элементов для каждого процесса
        MPI_DOUBLE,            // тип данных
        local_graph.data(),    // буфер приема
        rows_per_process * N,  // количество принимаемых элементов
        MPI_DOUBLE,            // тип данных
        0,                     // ранг отправителя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // буфер для хранения k-й строки (для broadcast)
    std::vector<double> k_row(N);
    
    // основной цикл алгоритма флойда-уоршелла
    // k - промежуточная вершина
    for (int k = 0; k < N; k++) {
        // определяем, какой процесс владеет k-й строкой
        int k_process = k / rows_per_process;
        
        // если текущий процесс владеет k-й строкой
        if (rank == k_process) {
            // локальный индекс k-й строки в массиве этого процесса
            int local_k = k % rows_per_process;
            
            // копируем k-ю строку в буфер для передачи
            for (int j = 0; j < N; j++) {
                k_row[j] = local_graph[local_k * N + j];
            }
        }
        
        // рассылаем k-ю строку всем процессам
        MPI_Bcast(
            k_row.data(),      // буфер с данными
            N,                 // количество элементов
            MPI_DOUBLE,        // тип данных
            k_process,         // процесс-отправитель
            MPI_COMM_WORLD     // коммуникатор
        );
        
        // каждый процесс обновляет свои строки
        for (int i = 0; i < rows_per_process; i++) {
            // глобальный индекс текущей строки
            int global_i = rank * rows_per_process + i;
            
            // для каждой вершины j
            for (int j = 0; j < N; j++) {
                // текущее расстояние от i до j
                double current_dist = local_graph[i * N + j];
                
                // расстояние через промежуточную вершину k
                double new_dist = local_graph[i * N + k] + k_row[j];
                
                // если новый путь короче, обновляем расстояние
                if (new_dist < current_dist) {
                    local_graph[i * N + j] = new_dist;
                }
            }
        }
    }
    
    // собираем результаты на главном процессе
    MPI_Gather(
        local_graph.data(),    // буфер отправки
        rows_per_process * N,  // количество отправляемых элементов
        MPI_DOUBLE,            // тип данных
        graph.data(),          // буфер приема (только на главном)
        rows_per_process * N,  // количество принимаемых элементов от каждого
        MPI_DOUBLE,            // тип данных
        0,                     // ранг получателя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // главный процесс выводит результаты
    if (rank == 0) {
        // измеряем время окончания
        double end_time = MPI_Wtime();
        
        std::cout << "\nМатрица кратчайших путей:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (graph[i * N + j] == INF) {
                    std::cout << std::setw(8) << "INF";
                } else {
                    std::cout << std::setw(8) << std::fixed << std::setprecision(1) << graph[i * N + j];
                }
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nВремя выполнения: " << end_time - start_time << " секунд" << std::endl;
    }
    
    // завершаем работу mpi окружения
    MPI_Finalize();
    
    return 0;
}
