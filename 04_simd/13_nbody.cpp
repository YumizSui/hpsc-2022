#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>


int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 mj = _mm256_load_ps(m);
  __m256 xj = _mm256_load_ps(x);
  __m256 yj = _mm256_load_ps(y);
  __m256 zero = _mm256_setzero_ps();

  for(int i=0; i<N; i++) {
    // calc rx
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 rx = _mm256_sub_ps(xj, xi);

    // calc rx
    __m256 yi = _mm256_set1_ps(y[i]);
    __m256 ry = _mm256_sub_ps(yj, yi);

    //calc r
    __m256 r2 = zero;
    r2 = _mm256_fmadd_ps(rx, rx, r2);
    r2 = _mm256_fmadd_ps(ry, ry, r2);
    __m256 inv_r = _mm256_rsqrt_ps(r2);
    __m256 mask = _mm256_cmp_ps(r2, zero, _CMP_GT_OQ);
    inv_r = _mm256_blendv_ps(zero, inv_r, mask);
    __m256 inv_r3 = _mm256_mul_ps(_mm256_mul_ps(inv_r, inv_r), inv_r);
    
    __m256 fxj = _mm256_mul_ps(_mm256_mul_ps(rx, mj), inv_r3);
    __m256 fyj = _mm256_mul_ps(_mm256_mul_ps(ry, mj), inv_r3);
    
    // reduction fx and fy
    __m256 fxi = _mm256_permute2f128_ps(fxj,fxj,1);
    fxi = _mm256_add_ps(fxi,fxj);
    fxi = _mm256_hadd_ps(fxi,fxi);
    fxi = _mm256_hadd_ps(fxi,fxi);

    __m256 fyi = _mm256_permute2f128_ps(fyj,fyj,1);
    fyi = _mm256_add_ps(fyi,fyj);
    fyi = _mm256_hadd_ps(fyi,fyi);
    fyi = _mm256_hadd_ps(fyi,fyi);
    _mm256_store_ps(fx, fxi);
    _mm256_store_ps(fy, fyi);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
