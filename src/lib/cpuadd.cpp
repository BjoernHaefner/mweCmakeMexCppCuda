
#ifndef _CPUADD_CPP_
#define _CPUADD_CPP_

#include "cpuadd.h"

void cpuadd(double *A, double *B, double *C, size_t Am, size_t An)
{
  for (size_t id = 0; id < Am * An; id++)
    C[id] = A[id] + B[id];
}

#endif
