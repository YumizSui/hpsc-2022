#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include "mpi.h"
using namespace std;
typedef vector<float> vect_f;

void sumcheck(vect_f& U, vect_f& V, vect_f& P, vect_f& B, const int N);

int main(int argc, char** argv) {

    const int N = 41;

    // mpi
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = rank * (N / size);
    int end = (rank + 1) * (N / size);
    int begin_i = begin;
    int end_i = end;
    if (rank == 0) {
        begin_i++;
    }
    if (rank == size-1) {
        end_i--;
        end+=N%size;
    }
    printf("%d: %d, %d\n", rank, begin, end);


    // initialize matrix
    vect_f U(N*N,0);
    vect_f V(N*N,0);
    vect_f P(N*N,0);
    vect_f B(N*N,0);
    const int Nt = 500, Nit = 50;
    const double dx = 2. / (N - 1);
    const double dy = 2. / (N - 1);
    const double dt = .01;
    const double rho = 1.;
    const double nu = .02;
    
    // calucularion navier-stokes
    auto tic = chrono::steady_clock::now();
    for (int n=0;n<Nt;n++) {
        // calcurate B
        for(int j=begin_i; j<end_i; j++) {
            for (int i=1;i<N-1;i++) {
                B[j*N+i] = rho * (1./dt*((U[j*N+(i+1)] - U[j*N+(i-1)]) / (2. * dx) + (V[(j+1)*N+i] - V[(j-1)*N+i]) / (2. * dy))
                            - pow((U[j*N+(i+1)] - U[j*N+(i-1)]) / (2. * dx), 2)
                            - 2. * ((U[(j+1)*N+i] - U[(j-1)*N+i]) / (2. * dy) * (V[j*N+(i+1)] - V[j*N+(i-1)]) / (2. * dx))
                            - pow((V[(j+1)*N+i] - V[(j-1)*N+i]) / (2. * dy), 2));
            }
        }
        MPI_Allgather( &B[begin*N], (end-begin)*N, MPI_FLOAT, &B[0], (end-begin)*N, MPI_FLOAT, MPI_COMM_WORLD);
        
        // Calcurate P for N times
        for (int it = 0;it<Nit;it++){
            vect_f Pn(P);
            for(int j=begin_i; j<end_i; j++) {
                for (int i=1;i<N-1;i++) {
                    P[j*N+i] = (pow(dy, 2) * (Pn[j*N+(i+1)] + Pn[j*N+(i-1)]) 
                                + pow(dx, 2) * (Pn[(j+1)*N+i] + Pn[(j-1)*N+i]) 
                                - B[j*N+i] * pow(dx, 2) * pow(dy, 2)) / (2. * (pow(dx, 2) + pow(dy, 2)));
                }
            }
            MPI_Allgather( &P[begin*N], (end-begin)*N, MPI_FLOAT, &P[0], (end-begin)*N, MPI_FLOAT, MPI_COMM_WORLD);
            for (int j=1;j<N-1;j++) {
                P[j*N+(N-1)] = P[j*N+(N-2)];
                P[j*N] = P[j*N+1];
                P[j] = P[1*N+j];
                P[(N-1)*N+j] = 0;
            }
        }

        // calcurate U and V
        vect_f Un(U);
        vect_f Vn(V);
        for(int j=begin_i; j<end_i; j++) {
            for (int i=1;i<N-1;i++) {
                U[j*N+i] = Un[j*N+i] - Un[j*N+i] * dt / dx * (Un[j*N+i] - Un[j*N+(i-1)])
                            - Un[j*N+i] * dt / dy * (Un[j*N+i] - Un[(j-1)*N+i])
                            - dt / (2. * rho * dx) * (P[j*N+(i+1)] - P[j*N+(i-1)])
                            + nu * dt / pow(dx, 2) * (Un[j*N+(i+1)] - 2. * Un[j*N+i] + Un[j*N+(i-1)])
                            + nu * dt / pow(dy, 2) * (Un[(j+1)*N+i] - 2. * Un[j*N+i] + Un[(j-1)*N+i]);
                V[j*N+i] = Vn[j*N+i] - Vn[j*N+i] * dt / dx * (Vn[j*N+i] - Vn[j*N+(i-1)])
                            - Vn[j*N+i] * dt / dy * (Vn[j*N+i] - Vn[(j-1)*N+i])
                            - dt / (2. * rho * dx) * (P[(j+1)*N+i] - P[(j-1)*N+i])
                            + nu * dt / pow(dx, 2) * (Vn[j*N+(i+1)] - 2. * Vn[j*N+i] + Vn[j*N+(i-1)])
                            + nu * dt / pow(dy, 2) * (Vn[(j+1)*N+i] - 2. * Vn[j*N+i] + Vn[(j-1)*N+i]);
            }
        }
        MPI_Allgather( &U[begin*N], (end-begin)*N, MPI_FLOAT, &U[0], (end-begin)*N, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather( &V[begin*N], (end-begin)*N, MPI_FLOAT, &V[0], (end-begin)*N, MPI_FLOAT, MPI_COMM_WORLD);
        for (int j=0;j<N;j++) {
            U[j*N]=0;
            U[j*N+(N-1)]=0;
            V[j*N]=0;
            V[j*N+(N-1)]=0;
            U[j]=0;
            V[j]=0;
            V[(N-1)*N+j]=0;
            U[(N-1)*N+j]=1;
        }
    }
    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();
    if (rank ==0){
        printf("%1.3lf sec\n", time);
        // sumcheck
        sumcheck(U, V, P, B, N);
    }
    MPI_Finalize();

    return 0;
}

// sumcheck
void sumcheck(vect_f& U, vect_f& V, vect_f& P, vect_f& B, const int N) {
    double sum_B, sum_P, sum_U, sum_V;
    sum_B = sum_P = sum_U = sum_V = 0.;
    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_B+=B[j*N+i];
        }
    }

    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_P+=P[j*N+i];
        }
    }

    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_U+=U[j*N+i];
        }
    }

    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_V+=V[j*N+i];
        }
    }
    printf("%2.6lf, %2.6lf, %2.6lf, %2.6lf\n", sum_B, sum_P, sum_U, sum_V);
}