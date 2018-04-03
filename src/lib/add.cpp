
#ifndef _ADD_CPP_
#define _ADD_CPP_

#include "add.h"

void add(double *A, double *B, double *C, size_t Am, size_t An)
{
  for (size_t id = 0; id < Am * An; id++)
    C[id] = A[id] + B[id];
}

#endif
