
#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_

void addkernel(double *A, double *B, double *C, mwSignedIndex Am, mwSignedIndex An)
{
  for (size_t id = 0; id < Am * An; id++)
    C[id] = A[id] + B[id];
}

#endif
