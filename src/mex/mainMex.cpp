/************************************************************************
 Sample CUDA MEX code written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

#ifndef _MAIN_MEX_CPP_
#define _MAIN_MEX_CPP_

#ifdef USE_MEX

/* system header */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

/* MEX header */
#include <mex.h> 
#include "matrix.h"

/* fixing error : identifier "IUnknown" is undefined" */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include "../lib/add.h"


/* MEX entry function */
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])

{
  printf("My Version of the code!!!!\n");
  
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

  add(A, B, C, Am, An);    
    
}
#endif//USE_MEX

#endif
