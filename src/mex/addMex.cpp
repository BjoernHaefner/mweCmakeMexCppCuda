/************************************************************************
 Sample CUDA MEX code written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

#ifdef USE_MEX

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

/* fixing error : identifier "IUnknown" is undefined" */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#ifdef USE_CUDA
/* includes CUDA kernel */
#include "../lib/gpuadd.cuh"
#else
#include "../lib/add.h"
#endif //USE_CUDA


/* MEX entry function */
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])

{
    double *A, *B, *C;
    size_t Am, An, Bm, Bn; 
    
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
    Am = mxGetM(prhs[0]);
    An = mxGetN(prhs[0]);    
    Bm = mxGetM(prhs[1]);
    Bn = mxGetN(prhs[1]);
    if ( Am != Bm || An != Bn) {
        mexErrMsgIdAndTxt("MATLAB:cudaAdd:sizemismatch",
                          "Input matrices must have the same size!");
    }

    /* allocate output */
    plhs[0] = mxCreateDoubleMatrix(Am, An, mxREAL);
    C = mxGetPr(plhs[0]);

#ifdef USE_CUDA
  gpuadd(A, B, C, Am, An); 
#else
	add(A, B, C, Am, An);    
#endif
    
}
#endif//USE_MEX
