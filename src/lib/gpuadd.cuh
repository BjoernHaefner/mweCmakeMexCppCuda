/************************************************************************
 Sample CUDA MEX kernel code written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

#ifndef _GPU_ADD_CUH_
#define _GPU_ADD_CUH_

void gpuadd(double *A, double *B, double *C, size_t Am, size_t An);

#endif
