#ifndef _ADD_CPP_
#define _ADD_CPP_

#include <stddef.h> //size_t
#include <iostream> //cout
#include <stdio.h> //printf

//Eigen
#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif //USE_EIGEN

#include "add.h"

#ifdef USE_CUDA
/* includes CUDA kernel */
#include "cuda/gpuadd.cuh"
#else
#include "cpuadd.h"
#endif //USE_CUDA

void add(double *A, double *B, double *C, size_t Am, size_t An)
{

#ifdef USE_EIGEN  
  std::cout << "But before, print a dummy matrix to show that Eigen is used in the Mex and binary case:" << std::endl;
  Eigen::MatrixXd my_dummy_matrix(2,2);
  my_dummy_matrix(0,0) = 3;
  my_dummy_matrix(1,0) = 2.5;
  my_dummy_matrix(0,1) = -1;
  my_dummy_matrix(1,1) = my_dummy_matrix(1,0) + my_dummy_matrix(0,1);
  std::cout << "my_dummy_matrix" << my_dummy_matrix << std::endl << std::endl;
#endif //USE_EIGEN

  std::cout << "Ok, now really add the two values:" << std::endl;
#ifdef USE_CUDA
  std::cout << "With CUDA support" << std::endl;
  gpuadd(A, B, C, Am, An); 
#else
  std::cout << "Without CUDA support" << std::endl;
	cpuadd(A, B, C, Am, An);    
#endif

}

#endif
