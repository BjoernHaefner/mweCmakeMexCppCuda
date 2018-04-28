#ifndef _CPUADD_CPP_
#define _CPUADD_CPP_

#include <stdio.h> //printf

#include "cpuadd.h"

void cpuadd(double *A, double *B, double *C, size_t Am, size_t An)
{
  printf("...Doing a for loop to add\n\n");
  for (size_t id = 0; id < Am * An; id++)
    C[id] = A[id] + B[id];
}

#endif
