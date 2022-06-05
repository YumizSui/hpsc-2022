
navier_openmp: navier_openmp.cpp
	g++ navier_openmp.cpp -fopenmp -O2
	./a.out 2> navier_openmp.log
navier_mpi: navier_mpi.cpp
	mpicxx navier_mpi.cpp
	mpirun -np 4 ./a.out 2> navier_mpi.log