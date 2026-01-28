#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

int main(int argc, char** argv) {
    // инициализация mpi окружения
    MPI_Init(&argc, &argv);
    
    // получаем общее количество процессов
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // получаем ранг текущего процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // размер системы уравнений (можно изменить)
    const int N = 8;
    
    // проверяем, что размер системы делится на количество процессов
    if (N % world_size != 0) {
        if (rank == 0) {
            std::cout << "Ошибка: размер матрицы должен делиться на количество процессов" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // вычисляем количество строк для каждого процесса
    int rows_per_process = N / world_size;
    
    // матрица коэффициентов (только на главном процессе)
    std::vector<double> A;
    
    // вектор правых частей (только на главном процессе)
    std::vector<double> b;
    
    // вектор решений (только на главном процессе)
    std::vector<double> x(N);
    
    // измеряем время начала
    double start_time = MPI_Wtime();
    
    // главный процесс создает систему уравнений
    if (rank == 0) {
        // выделяем память для матрицы
        A.resize(N * N);
        b.resize(N);
        
        // инициализируем генератор случайных чисел
        srand(42); // фиксированный seed для воспроизводимости
        
        // создаем систему с диагональным преобладанием (для устойчивости)
        for (int i = 0; i < N; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (i != j) {
                    // недиагональные элементы
                    A[i * N + j] = (rand() % 10) + 1;
                    row_sum += fabs(A[i * N + j]);
                } else {
                    // диагональный элемент (заполним позже)
                    A[i * N + j] = 0;
                }
            }
            // делаем диагональный элемент больше суммы остальных
            A[i * N + i] = row_sum + (rand() % 10) + 10;
            
            // правая часть
            b[i] = (rand() % 20) + 1;
        }
        
        std::cout << "Система создана. Размер: " << N << "x" << N << std::endl;
        std::cout << "Количество процессов: " << world_size << std::endl;
    }
    
    // каждый процесс выделяет память для своей части матрицы
    // размер: rows_per_process строк по N элементов в каждой
    std::vector<double> local_A(rows_per_process * N);
    
    // каждый процесс выделяет память для своей части вектора b
    std::vector<double> local_b(rows_per_process);
    
    // распределяем строки матрицы между процессами
    MPI_Scatter(
        A.data(),              // буфер отправки
        rows_per_process * N,  // количество элементов для каждого процесса
        MPI_DOUBLE,            // тип данных
        local_A.data(),        // буфер приема
        rows_per_process * N,  // количество принимаемых элементов
        MPI_DOUBLE,            // тип данных
        0,                     // ранг отправителя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // распределяем элементы вектора b между процессами
    MPI_Scatter(
        b.data(),              // буфер отправки
        rows_per_process,      // количество элементов для каждого процесса
        MPI_DOUBLE,            // тип данных
        local_b.data(),        // буфер приема
        rows_per_process,      // количество принимаемых элементов
        MPI_DOUBLE,            // тип данных
        0,                     // ранг отправителя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // буфер для хранения текущей строки (для broadcast)
    std::vector<double> pivot_row(N + 1); // N элементов матрицы + 1 элемент вектора b
    
    // прямой ход метода гаусса
    for (int k = 0; k < N; k++) {
        // определяем, какой процесс владеет k-й строкой
        int pivot_process = k / rows_per_process;
        
        // если текущий процесс владеет k-й строкой
        if (rank == pivot_process) {
            // локальный индекс k-й строки в массиве этого процесса
            int local_row = k % rows_per_process;
            
            // копируем k-ю строку в буфер для передачи
            for (int j = 0; j < N; j++) {
                pivot_row[j] = local_A[local_row * N + j];
            }
            pivot_row[N] = local_b[local_row]; // добавляем элемент из b
        }
        
        // рассылаем k-ю строку всем процессам
        MPI_Bcast(
            pivot_row.data(),  // буфер с данными
            N + 1,             // количество элементов
            MPI_DOUBLE,        // тип данных
            pivot_process,     // процесс-отправитель
            MPI_COMM_WORLD     // коммуникатор
        );
        
        // каждый процесс обрабатывает свои строки
        for (int i = 0; i < rows_per_process; i++) {
            // глобальный индекс текущей строки
            int global_row = rank * rows_per_process + i;
            
            // обрабатываем только строки ниже k-й
            if (global_row > k) {
                // вычисляем множитель для вычитания
                double factor = local_A[i * N + k] / pivot_row[k];
                
                // вычитаем k-ю строку, умноженную на factor
                for (int j = k; j < N; j++) {
                    local_A[i * N + j] -= factor * pivot_row[j];
                }
                
                // обновляем правую часть
                local_b[i] -= factor * pivot_row[N];
            }
        }
    }
    
    // собираем обработанную матрицу на главном процессе
    MPI_Gather(
        local_A.data(),        // буфер отправки
        rows_per_process * N,  // количество отправляемых элементов
        MPI_DOUBLE,            // тип данных
        A.data(),              // буфер приема (только на главном)
        rows_per_process * N,  // количество принимаемых элементов от каждого
        MPI_DOUBLE,            // тип данных
        0,                     // ранг получателя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // собираем обработанный вектор b на главном процессе
    MPI_Gather(
        local_b.data(),        // буфер отправки
        rows_per_process,      // количество отправляемых элементов
        MPI_DOUBLE,            // тип данных
        b.data(),              // буфер приема
        rows_per_process,      // количество принимаемых элементов от каждого
        MPI_DOUBLE,            // тип данных
        0,                     // ранг получателя
        MPI_COMM_WORLD         // коммуникатор
    );
    
    // обратный ход выполняет только главный процесс
    if (rank == 0) {
        // идем от последней строки к первой
        for (int i = N - 1; i >= 0; i--) {
            // начинаем с правой части
            x[i] = b[i];
            
            // вычитаем известные x[j] * A[i][j] для j > i
            for (int j = i + 1; j < N; j++) {
                x[i] -= A[i * N + j] * x[j];
            }
            
            // делим на диагональный элемент
            x[i] /= A[i * N + i];
        }
        
        // измеряем время окончания
        double end_time = MPI_Wtime();
        
        // выводим решение
        std::cout << "\nРешение системы уравнений:" << std::endl;
        for (int i = 0; i < N; i++) {
            std::cout << "x[" << i << "] = " << std::fixed << std::setprecision(6) << x[i] << std::endl;
        }
        std::cout << "\nВремя выполнения: " << end_time - start_time << " секунд" << std::endl;
    }
    
    // завершаем работу mpi окружения
    MPI_Finalize();
    
    return 0;
}
