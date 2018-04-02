/************************************************************************
 Sample CUDA MEX code written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

/* system header */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#ifdef USE_MEX
/* MEX header */
#include <mex.h> 
#include "matrix.h"
#endif //USE_MEX

#ifdef USE_CUDA
/* nVIDIA CUDA header */
#include <cuda.h> 
#endif //USE_CUDA

/* fixing error : identifier "IUnknown" is undefined" */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#ifdef USE_CUDA
/* includes CUDA kernel */
#include "gpuaddkernel.cuh"
#else
#include "addkernel.h"
#endif //USE_CUDA


#ifdef USE_MEX
/* MEX entry function */
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])

{
    double *A, *B, *C;
    mwSignedIndex Am, An, Bm, Bn; 
    
    /* argument check */
    if ( nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:inputmismatch",
                          "Input arguments must be 2!");
    }
    if ( nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:outputmismatch",
                          "Output arguments must be 1!");
    }

    A = mxGetPr(prhs[0]); 
    B = mxGetPr(prhs[1]);

    /* matrix size */
    Am = (mwSignedIndex)mxGetM(prhs[0]);
    An = (mwSignedIndex)mxGetN(prhs[0]);    
    Bm = (mwSignedIndex)mxGetM(prhs[1]);
    Bn = (mwSignedIndex)mxGetN(prhs[1]);
    if ( Am != Bm || An != Bn) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:sizemismatch",
                          "Input matrices must have the same size!");
    }

    /* allocate output */
    plhs[0] = mxCreateDoubleMatrix(Am, An, mxREAL);
    C = mxGetPr(plhs[0]);

#ifdef USE_CUDA
    /* set GPU grid & block configuration */
    cudaDeviceProp deviceProp;
    memset( &deviceProp, 0, sizeof(deviceProp));
    if( cudaSuccess != cudaGetDeviceProperties(&deviceProp,0)){
        mexPrintf( "\n%s", cudaGetErrorString(cudaGetLastError()));
        return;
    }

	  dim3 dimGridImg(8,1,1);
    dim3 dimBlockImg(1,64,1);
	
    /* allocate device memory for matrices */
    double *d_A = NULL;
    cudaMalloc( (void**) &d_A, Am * An * sizeof(double)) ;
	cudaMemcpy( d_A, A, Am * An * sizeof(double), cudaMemcpyHostToDevice) ;
    double *d_B = NULL;
    cudaMalloc( (void**) &d_B, Bm * Bn * sizeof(double)) ;
	cudaMemcpy( d_B, B, Bm * Bn * sizeof(double), cudaMemcpyHostToDevice) ;
    double *d_C = NULL;
    cudaMalloc( (void**) &d_C, Am * An * sizeof(double)) ;
    
	/* call GPU kernel for addition */
	gpuaddkernel<<< dimGridImg, dimBlockImg >>>(d_A, d_B, d_C, Am, An);
	cudaThreadSynchronize();
    
    /* copy result from device */
	cudaMemcpy( C, d_C, Am * An * sizeof(double), cudaMemcpyDeviceToHost) ;

    /* free GPU memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#else
	addkernel(A, B, C, Am, An);    
#endif
    
}
#endif//USE_MEX
