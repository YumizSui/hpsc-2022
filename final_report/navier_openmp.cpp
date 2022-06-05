#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
using namespace std;
typedef vector<vector<float>> matrix;

void sumcheck(matrix& U, matrix& V, matrix& P, matrix& B, const int N);

void navier_stokes(matrix& U, matrix& V, matrix& P, matrix& B, const int N) {
    const int Nt = 500, Nit = 50;
    const double dx = 2. / (N - 1);
    const double dy = 2. / (N - 1);
    const double dt = .01;
    const double rho = 1.;
    const double nu = .02;

#pragma omp parallel
    for (int n=0;n<Nt;n++) {
#pragma omp for collapse(2)
        for (int j=1;j<N-1; j++) {
            for (int i=1;i<N-1;i++) {
                B[j][i] = rho * (1./dt*((U[j][i+1] - U[j][i-1]) / (2. * dx) + (V[j+1][i] - V[j-1][i]) / (2. * dy))
                            - pow((U[j][i+1] - U[j][i-1]) / (2. * dx), 2)
                            - 2. * ((U[j+1][i] - U[j-1][i]) / (2. * dy) * (V[j][i+1] - V[j][i-1]) / (2. * dx))
                            - pow((V[j+1][i] - V[j-1][i]) / (2. * dy), 2));
                // B[j][i] = 1./dt*((U[j][i+1] - U[j][i-1]) / (2. * dx) + (V[j+1][i] - V[j-1][i]) / (2. * dy));
            }
        }
        for (int it = 0;it<Nit;it++){
            matrix Pn(P);
#pragma omp for collapse(2)
            for (int j=1;j<N-1;j++) {
                for (int i=1;i<N-1;i++) {
                    P[j][i] = (pow(dy, 2) * (Pn[j][i+1] + Pn[j][i-1]) 
                                + pow(dx, 2) * (Pn[j+1][i] + Pn[j-1][i]) 
                                - B[j][i] * pow(dx, 2) * pow(dy, 2)) / (2. * (pow(dx, 2) + pow(dy, 2)));
                }
            }
#pragma omp for
            for (int j=0;j<N;j++) {
                P[j][N-1] = P[j][N-2];
                P[j][0] = P[j][1];
                P[0][j] = P[1][j];
                P[N-1][j] = 0;
            }
        }
        matrix Un(U);
        matrix Vn(V);
#pragma omp for collapse(2)
        for (int j=1;j<N-1;j++) {
            for (int i=1;i<N-1;i++) {
                U[j][i] = Un[j][i] - Un[j][i] * dt / dx * (Un[j][i] - Un[j][i - 1])
                            - Un[j][i] * dt / dy * (Un[j][i] - Un[j - 1][i])
                            - dt / (2. * rho * dx) * (P[j][i+1] - P[j][i-1])
                            + nu * dt / pow(dx, 2) * (Un[j][i+1] - 2. * Un[j][i] + Un[j][i-1])
                            + nu * dt / pow(dy, 2) * (Un[j+1][i] - 2. * Un[j][i] + Un[j-1][i]);
                V[j][i] = Vn[j][i] - Vn[j][i] * dt / dx * (Vn[j][i] - Vn[j][i - 1])
                            - Vn[j][i] * dt / dy * (Vn[j][i] - Vn[j - 1][i])
                            - dt / (2. * rho * dx) * (P[j+1][i] - P[j-1][i])
                            + nu * dt / pow(dx, 2) * (Vn[j][i+1] - 2. * Vn[j][i] + Vn[j][i-1])
                            + nu * dt / pow(dy, 2) * (Vn[j+1][i] - 2. * Vn[j][i] + Vn[j-1][i]);
            }
        }
#pragma omp for
        for (int j=0;j<N;j++) {
            U[j][0]=0;
            U[j][N-1]=0;
            V[j][0]=0;
            V[j][N-1]=0;
            U[0][j]=0;
            V[0][j]=0;
            V[N-1][j]=0;
            U[N-1][j]=1;
        }
    }
}

int main() {
    // initialize matrix
    const int N = 40;
    matrix U(N,vector<float>(N,0));
    matrix V(N,vector<float>(N,0));
    matrix P(N,vector<float>(N,0));
    matrix B(N,vector<float>(N,0));


    // calucularion navier-stokes
    auto tic = chrono::steady_clock::now();
    navier_stokes(U, V, P, B, N);
    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();
    printf("%1.3lf sec\n", time);

    // sumcheck
    sumcheck(U, V, P, B, N);
    return 0;
}

// sumcheck
void sumcheck(matrix& U, matrix& V, matrix& P, matrix& B, const int N) {
    double sum_B, sum_P, sum_U, sum_V;
    sum_B = sum_P = sum_U = sum_V = 0.;
    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_B+=B[j][i];
        }
    }

    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_P+=P[j][i];
        }
    }

    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_U+=U[j][i];
        }
    }

    for (int j=0;j<N; j++) {
        for (int i=0;i<N;i++) {
            sum_V+=V[j][i];
        }
    }
    printf("%2.6lf, %2.6lf, %2.6lf, %2.6lf\n", sum_B, sum_P, sum_U, sum_V);
}