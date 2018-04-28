#ifndef _GPU_ADD_CU_
#define _GPU_ADD_CU_

#include <stdio.h> //size_t
#include <math.h> //fabs
#include <cstring> //memcpy

//CUDA libraries
#include "cublas_v2.h"

//HEADER file
#include "gpuadd.cuh"

//the array index of a matrix element in row “r” and column “c” can be computed via the following macro
#define IDX2C(r,c,rows) (((c)*(rows))+(r))

__global__ void gpuaddkernel(double *d_A, double *d_B, double *d_C, size_t Am, size_t An)
{
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int ind = x + An*y; //derive linear index
  if (x<An && y<Am) d_C[ind] = d_A[ind] + d_B[ind];
}

void gpuaddcublas(double *A, double *B, double *C, size_t Am, size_t An)
{

    cudaError_t cudaStat;    
    cublasStatus_t stat;
    cublasHandle_t handle;

    double *d_A, *d_C;
    if (!A || !B || !C) {
        printf ("Error in gpuaddcublas: input failed\n");
        return;
    }
    
    cudaStat = cudaMalloc ((void**)&d_A, Am*An*sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return;
    }
    cudaStat = cudaMalloc ((void**)&d_C, Am*An*sizeof(double));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return;
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return;
    }

    //for (size_t ii=0; ii<Am*An; ii++) printf("A=%f\n",A[ii]); 

    stat = cublasSetVector(Am*An, sizeof(double), (void*)A, 1, (void*)d_A, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download for A failed: %d\n", stat);
        cudaFree (d_A);
        cublasDestroy(handle);
        return;
    }
    stat = cublasSetVector(Am*An, sizeof(double), (void*)B, 1, (void*)d_C, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download for B failed\n");
        cudaFree (d_A);
        cudaFree (d_C);
        cublasDestroy(handle);
        return;
    }

  //add two matrices (represented as vectors)
    double alpha = 1.;
    stat =  cublasAxpyEx (handle,Am*An, &alpha, CUDA_R_64F, (void *)d_A, CUDA_R_64F, 1, (void *)d_C, CUDA_R_64F,  1, CUDA_R_64F);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data adding failed\n");
        cudaFree (d_A);
        cudaFree (d_C);
        cublasDestroy(handle);
        return;
    }

    stat = cublasGetVector(Am*An, sizeof(double), (void *)d_C, 1, (void*)C, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed\n");
        cudaFree (d_A);
        cudaFree (d_C);
        cublasDestroy(handle);
        return;
    }
   cudaFree (d_A);
   cudaFree (d_C);
   cublasDestroy(handle);
   return;

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


//First add two pointers with specified kernel

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
	cudaDeviceSynchronize();
    
  /* copy result from device */
  double *C_kernel = new double [Am*An];
	cudaMemcpy( C_kernel, d_C, Am * An * sizeof(double), cudaMemcpyDeviceToHost) ;


  /* free GPU memory */
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

//Now add pointers with cublas
  double *C_cublas = new double [Am*An];
	gpuaddcublas(A, B, C_cublas, Am, An);

//Compare results
bool equal = true;
for (size_t ii=0; ii<Am*An; ii++)
  if (!(fabs(C_kernel[ii]-C_cublas[ii])<0.001))
    equal = false;

if (equal)
{
  printf("Yeah, both arrays have the same values\n");
  std::memcpy( (void*)C, (void*) C_kernel, Am * An * sizeof(double) );
  free(C_kernel);
  free(C_cublas);
}
else
{
  printf("Oh no, cublas and your own kernel differ too much in results.\n");
  printf("Copy kernel results now, but handle with caution.\n");
  std::memcpy( (void*)C, (void*) C_kernel, Am * An * sizeof(double) );
  free(C_kernel);
  free(C_cublas);
}

}

#endif
