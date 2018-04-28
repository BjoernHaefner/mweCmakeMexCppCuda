CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(mweAdd) #project name

##################ADJUST##############################
######################################################
#what shall be build?
SET(MEXF TRUE) #generate mex file for matlab usage
SET(EXEF TRUE) #generate executable to call code from commandline
SET(CUDA FALSE) #use cuda for speedup if applicable

#set cpp and cuda files
SET(CPP_MAIN_FILES src/main.cpp) #main file(s) for cpp
SET(MEX_MAIN_FILES src/mex/mainMex.cpp ) #main file(s) for mex (mex entrance function)

SET(COMMON_SRC_FILES src/lib/add.cpp src/lib/add.h ) #files needed for both, cpp and cuda built

SET(CU_SRC_FILES src/lib/cuda/gpuadd.cu src/lib/cuda/gpuadd.cuh ) #cuda files (e.g. kernels)
SET(CPP_SRC_FILES src/lib/cpuadd.cpp src/lib/cpuadd.h ) #cpp files needed for built with non-cuda support only

#set libraries
SET(OPENCV_LIB OpenCV) #used for non-mex build only
SET(CUDA_LIBS CUDA_CUBLAS_LIBRARIES CUDA_CUFFT_LIBRARIES) #used for cuda build only
SET(EIGEN_LIB Eigen3)#used for mex, non-mex and cuda build

##################SETUP###############################
######################################################
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) #output for executable i.e., EXEF=TRUE
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)    #output for mex-file i.e., MEXF=TRUE

SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) #add modules e.g., FindMatlab.cmake

MESSAGE(STATUS "MEXF=" ${MEXF})
MESSAGE(STATUS "EXEF=" ${EXEF})
MESSAGE(STATUS "CUDA=" ${CUDA})


##################FIND OTHER LIBRARIES################
######################################################
FIND_PACKAGE( ${OPENCV_LIB} REQUIRED)
FIND_PACKAGE( ${EIGEN_LIB}  REQUIRED)

##################FIND MATLAB#########################
######################################################
if(${MEXF})
  #not sure if the following two definitions are necessary
#  ADD_DEFINITIONS(/DMATLAB_MEX_FILE) #define matlab macros
#  ADD_DEFINITIONS(/DMX_COMPAT_32)

  FIND_PACKAGE(Matlab REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if matlab is not used
  ADD_DEFINITIONS(-DUSE_MEX)

  # set up matlab libraries
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

  #use correct suffix depending on OS
  if(WIN32) # 32-bit or 64-bit mex
      if (CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw64 )
      else(CMAKE_CL_64)
        SET( MEX_SUFFIX .mexw32 )
      endif(CMAKE_CL_64)
  else(WIN32)
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexa64 )
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
        SET( MEX_SUFFIX .mexglx )
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif(WIN32)
endif(${MEXF})

##################FIND CUDA###########################
######################################################
if(${CUDA})

  FIND_PACKAGE(CUDA REQUIRED)

  #add definition USE_MEX for C++ code, to be able to use preprocessor if CUDA is not used
  ADD_DEFINITIONS(-DUSE_CUDA)

  # set SRC_FILES using the common cpp files and cu source files
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CU_SRC_FILES})

  #to correctly include the above cuda libraries, a splitting must take place
  foreach(libname ${CUDA_LIBS})
    LIST(APPEND CUDA_LIBS_SEP ${${libname}})
  endforeach(${CUDA_LIBS_SEP})

  #compiler flags for cuda
  SET(CUDA_NVCC_FLAGS
	-Xcompiler #Explicitly specify the language for the input files, rather than letting the compiler choose a default based on the file name suffix. Allowed values for this option: c, c++, cu. 
	-fPIC
	-use_fast_math
	-gencode=arch=compute_20,code="sm_20,compute_20"
	--ptxas-options=-v
	#--device-debug #possibility to debug device code; uncomment for speedup
	#--debug #possibility to debug host code; uncomment for speedup
	-O3 #specify optimization level for host code
	-Wno-deprecated-gpu-targets #to suppress warning for deprecated architectures
	)

else(${CUDA})#if cuda is not used 
  #set SRC_FILES using the common cpp files and cpp source files
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CPP_SRC_FILES} )

endif(${CUDA})

######################################################
##################BUILD FILES#########################
######################################################

######################MEX#############################
######################################################
if(${MEXF})

  #if mex lib is build, add suffix MEX to lib file
  SET(MEX_LIB_NAME "${PROJECT_NAME}MEX")
  
  if(${CUDA})
    CUDA_COMPILE(${MEX_LIB_NAME} ${MEX_MAIN_FILES} ${SRC_FILES} SHARED)   
  endif(${CUDA})

  ADD_LIBRARY(${MEX_LIB_NAME} SHARED ${${MEX_LIB_NAME}} ${MEX_MAIN_FILES} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)

  MESSAGE(STATUS "LINKING CUDA LIBRARIES: " ${CUDA_LIBS_SEP})
  TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                        ${MATLAB_LIBRARIES}
                        Eigen3::Eigen #in any case link to Eigen library for the mex file
                        )
  if(${CUDA})
    TARGET_LINK_LIBRARIES(${MEX_LIB_NAME}
                          ${CUDA_LIBRARIES} #Cudart library.
                          ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                          )
  endif(${CUDA})

  SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES PREFIX "" SUFFIX ${MEX_SUFFIX})
  if(${CUDA})
    SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif(${CUDA})

endif(${MEXF})


##################EXECUTABLE##########################
######################################################
if(${EXEF}) #now take care of the executabel

  if(${CUDA}) #if cuda is used
    CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES} )
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                          ${CUDA_LIBRARIES} #Cudart library.
                          ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
                          ${OpenCV_LIBS} #load opencv libs
                          Eigen3::Eigen #link to eigen library for binary with cuda support
                          )
  else(${CUDA})#is cuda is not used
    ADD_EXECUTABLE(${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}
                          ${OpenCV_LIBS} #load opencv libs
                          Eigen3::Eigen #link to eigen library for binary without cuda support
                          )
endif(${CUDA})

endif(${EXEF})



