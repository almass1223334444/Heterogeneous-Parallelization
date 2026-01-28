# компилятор для mpi программ на c++
CXX = mpic++

# флаги компиляции: включаем оптимизацию и предупреждения
CXXFLAGS = -O2 -Wall

# список всех исполняемых файлов
TARGETS = task1 task2 task3 task4

# правило по умолчанию: собрать все программы
all: $(TARGETS)

# правило для компиляции task1
task1: task1.cpp
	$(CXX) $(CXXFLAGS) task1.cpp -o task1

# правило для компиляции task2
task2: task2.cpp
	$(CXX) $(CXXFLAGS) task2.cpp -o task2

# правило для компиляции task3
task3: task3.cpp
	$(CXX) $(CXXFLAGS) task3.cpp -o task3

# правило для компиляции task4
task4: task4.cpp
	$(CXX) $(CXXFLAGS) task4.cpp -o task4

# правило для очистки скомпилированных файлов
clean:
	rm -f $(TARGETS)

# правило для запуска task1 с 4 процессами
run1:
	mpirun -np 4 ./task1

# правило для запуска task2 с 4 процессами
run2:
	mpirun -np 4 ./task2

# правило для запуска task3 с 4 процессами
run3:
	mpirun -np 4 ./task3

# правило для запуска task4 с 4 процессами
run4:
	mpirun -np 4 ./task4

# указываем, что эти цели не являются файлами
.PHONY: all clean run1 run2 run3 run4
