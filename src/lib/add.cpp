#ifndef _ADD_CPP_
#define _ADD_CPP_

#include <stddef.h> //size_t
#include <iostream> //cout
#include <stdio.h> //printf

//Eigen
#include <Eigen/Dense>

#include "add.h"

#ifdef USE_CUDA
/* includes CUDA kernel */
#include "cuda/gpuadd.cuh"
#else
#include "cpuadd.h"
#endif //USE_CUDA

void add(double *A, double *B, double *C, size_t Am, size_t An)
{
  
  printf("But before, print a dummy matrix to show that Eigen is used in the Mex and binary case:\n");
  Eigen::MatrixXd my_dummy_matrix(2,2);
  my_dummy_matrix(0,0) = 3;
  my_dummy_matrix(1,0) = 2.5;
  my_dummy_matrix(0,1) = -1;
  my_dummy_matrix(1,1) = my_dummy_matrix(1,0) + my_dummy_matrix(0,1);
  std::cout << "my_dummy_matrix" << my_dummy_matrix << std::endl << std::endl;

  printf("Ok, now really add the two values:\n");
#ifdef USE_CUDA
  printf("With CUDA support\n");
  gpuadd(A, B, C, Am, An); 
#else
  printf("Without CUDA support\n");
	cpuadd(A, B, C, Am, An);    
#endif

}

#endif
