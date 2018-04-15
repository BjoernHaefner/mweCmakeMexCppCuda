
#ifndef _MAIN_CPP_
#define _MAIN_CPP_

#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */

#ifdef USE_CUDA
/* includes CUDA kernel */
#include "lib/gpuadd.cuh"

#else //USE_CUDA
/* includes CPP 'kernel' */
#include "lib/add.h"

#endif //USE_CUDA

int main(int argc, char *argv[])
{

  if (argc != 3) return -1;

  double *A = (double*) malloc (1);
  double *B = (double*) malloc (1);
  double *C = (double*) malloc (1);

  size_t Am = 1, An = 1;

  A[0] = atof(argv[1]);
  B[0] = atof(argv[2]);

  printf("argc = %d\n", argc);
  for(int ndx = 0; ndx != argc; ++ndx)
    printf("argv[%d] --> %s\n", ndx,argv[ndx]);
  printf("argv[argc] = %p\n", (void*)argv[argc]);




#ifdef USE_CUDA
  gpuadd(A, B, C, Am, An); 
#else
	add(A, B, C, Am, An);    
#endif

  printf("A[0] + B[0] = %f + %f = %f\n", A[0], B[0], C[0]);

  free(A);
  free(B);
  free(C);

  return 0;

}

#endif
