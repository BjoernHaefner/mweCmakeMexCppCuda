/************************************************************************
 Sample CUDA MEX kernel code written by Fang Liu (leoliuf@gmail.com).
************************************************************************/

#ifndef _ADD_KERNEL_GPU_H_
#define _ADD_KERNEL_GPU_H_

__global__ void
gpuaddkernel(double *d_A, double *d_B, double *d_C, mwSignedIndex Am, mwSignedIndex An)
{
    /* index */
	unsigned int tid = blockIdx.x * blockDim.y + threadIdx.y; /* thread id in matrix*/
	/* strip */
	unsigned int strip = gridDim.x * blockDim.y;

	while (1) {
	 if (tid  < Am * An){
		d_C[tid] = d_A[tid] + d_B[tid];
	 }
	 else{
	    break;
	 }
	 tid += strip;
	}
}

#endif