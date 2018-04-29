#ifndef _CPUADD_CPP_
#define _CPUADD_CPP_

#include <iostream> //cout

#include "cpuadd.h"

void cpuadd(double *A, double *B, double *C, size_t Am, size_t An)
{
  std::cout << "...Doing a for loop to add" << std::endl << std::endl;
  for (size_t id = 0; id < Am * An; id++)
    C[id] = A[id] + B[id];
}

#endif
