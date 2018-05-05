
#ifndef _MAIN_CPP_
#define _MAIN_CPP_

#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */

//OpenCV
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif //USE_OPENCV

//own code
#include "lib/add.h"

int main(int argc, char *argv[])
{

  //print input
  printf("The commandline input is:\n");
  for(int ndx = 0; ndx < argc; ++ndx)
    printf("argv[%d] --> %s\n", ndx,argv[ndx]);

  //add the two values  
  double *A = (double*) malloc (1);
  double *B = (double*) malloc (1);
  double *C = (double*) malloc (1);

  size_t Am = 1, An = 1;

  A[0] = atof(argv[1]);
  B[0] = atof(argv[2]);

  
  printf("\nNow add the two values stored in argv[1] and argv[2]:\n");
  add(A, B, C, Am, An);  

  printf("Now you can check for yourself:\n");
  printf("A[0] + B[0] = %f + %f = %f\n\n", A[0], B[0], C[0]);

  free(A);
  free(B);
  free(C);

#ifdef USE_OPENCV
  //show the input image
  printf("Last, we quickly read an image to check if opencv works in the binary version...\n");
  cv::Mat image;
  image = cv::imread( argv[3], 1 );

  if ( !image.data )
  {
      printf("No image data \n");
      return -1;
  }
  else
  {
    printf("Could read image. OpenCV is working\n");
  }
#endif //USE_OPENCV
  return 0;

}

#endif
