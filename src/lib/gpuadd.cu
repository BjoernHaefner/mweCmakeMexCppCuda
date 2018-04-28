/************************************************************************
 Sample CUDA MEX kernel code written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

#ifndef _GPU_ADD_CU_
#define _GPU_ADD_CU_

#include <stdio.h>

#include "gpuadd.cuh"

__global__ void gpuaddkernel(double *d_A, double *d_B, double *d_C, size_t Am, size_t An)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int ind = x + An*y; //derive linear index
  if (x<An && y<Am) d_C[ind] = d_A[ind] + d_B[ind];
}

void gpuadd(double *A, double *B, double *C, size_t Am, size_t An)
{

  /* set GPU grid & block configuration */
  cudaDeviceProp deviceProp;
  memset( &deviceProp, 0, sizeof(deviceProp));
  if( cudaSuccess != cudaGetDeviceProperties(&deviceProp,0)){
    printf( "\n%s", cudaGetErrorString(cudaGetLastError()));
    return;
  }

  dim3 block = dim3(32,8,1); // 32*8*1 = 256 threads per block
  // ensure enough blocks to cover w * h elements (round up)
  dim3 grid = dim3( ( An + block.x -1 ) / block.x, ( Am + block.y - 1 ) / block.y, 1);
	
  /* allocate device memory for matrices */
  double *d_A = NULL;
  cudaMalloc( (void**) &d_A, Am * An * sizeof(double)) ;
	cudaMemcpy( d_A, A, Am * An * sizeof(double), cudaMemcpyHostToDevice) ;
  double *d_B = NULL;
  cudaMalloc( (void**) &d_B, Am * An * sizeof(double)) ;
	cudaMemcpy( d_B, B, Am * An * sizeof(double), cudaMemcpyHostToDevice) ;
  double *d_C = NULL;
  cudaMalloc( (void**) &d_C, Am * An * sizeof(double)) ;
    
	/* call GPU kernel for addition */
	gpuaddkernel<<< grid, block >>>(d_A, d_B, d_C, Am, An);
	cudaThreadSynchronize();
    
  /* copy result from device */
	cudaMemcpy( C, d_C, Am * An * sizeof(double), cudaMemcpyDeviceToHost) ;


  /* free GPU memory */
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

}

#endif
