#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void prepareBucket(int *key, int *bucket){
  int i = threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void countKey(int *key, int *bucket){
  int i = threadIdx.x;
  int j = 0;
  // calc start index j
  for (int k=0; k<i; k++) {
    j += bucket[k];
  }

  // count key
  for (; bucket[i]>0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  
  prepareBucket<<<1,n>>>(key, bucket);
  countKey<<<1,range>>>(key, bucket);

  cudaDeviceSynchronize();
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
}