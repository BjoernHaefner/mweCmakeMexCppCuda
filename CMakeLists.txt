CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(mweAdd)

##################ADJUST##############################
######################################################
SET(MEXF TRUE) #generate mex file for matlab usage
SET(EXEF TRUE) #generate executable to call code from commandline
SET(CUDA TRUE) #use cuda for speedup if applicable


SET(CPP_MAIN_FILES src/main.cpp) #main file(s) for cpp
SET(MEX_MAIN_FILES src/mex/mainMex.cpp ) #main file(s) for mex (mex entrance function)

SET(COMMON_SRC_FILES  ) #files needed for both, cpp and cuda built

SET(CU_SRC_FILES src/lib/gpuadd.cu ) #cuda files (e.g. kernels)
SET(CPP_SRC_FILES src/lib/add.cpp ) #cpp files needed for built with non-cuda support only

##################SETUP###############################
######################################################
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) #output for executable i.e., EXEF=TRUE
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)    #output for mex-file i.e., MEXF=TRUE

#add modules e.g., FindMatlab.cmake
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

##################FIND MATLAB#########################
######################################################
if(${MEXF})
  #ADD_DEFINITIONS(/DMATLAB_MEX_FILE) #define matlab macros
  #ADD_DEFINITIONS(/DMX_COMPAT_32)

  FIND_PACKAGE(Matlab)

  ADD_DEFINITIONS(-DUSE_MEX)

  #use correct suffix
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
  FIND_PACKAGE(CUDA)
  ADD_DEFINITIONS(-DUSE_CUDA)
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CU_SRC_FILES} )
  #compiler flags
  SET(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0 -Wno-deprecated-gpu-targets)
else(${CUDA})
  SET(SRC_FILES ${COMMON_SRC_FILES} ${CPP_SRC_FILES} )
endif(${CUDA})

######################################################
##################BUILD FILES#########################
######################################################

######################MEX#############################
######################################################
if(${MEXF})#build mex file with cuda support
  # compile cuda mex for shader models 2.0 as default, support double
  SET(MEX_LIB_NAME "${PROJECT_NAME}MEX")
  
  # set up cuda & matlab libraries
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})
  ADD_LIBRARY(${MEX_LIB_NAME} SHARED ${MEX_MAIN_FILES} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
  TARGET_LINK_LIBRARIES(${MEX_LIB_NAME} ${MATLAB_LIBRARIES})
  SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES PREFIX "" SUFFIX ${MEX_SUFFIX})
  if(${CUDA})
    CUDA_COMPILE(${MEX_LIB_NAME} ${MEX_MAIN_FILES} ${SRC_FILES} SHARED)
    TARGET_LINK_LIBRARIES(${MEX_LIB_NAME} ${CUDA_LIBRARIES})
    SET_TARGET_PROPERTIES(${MEX_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif(${CUDA})
endif(${MEXF})


##################EXECUTABLE##########################
######################################################
if(${EXEF})#build executable
  if(${CUDA})
    CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES} )
  else(${CUDA})
    ADD_EXECUTABLE(${PROJECT_NAME} ${CPP_MAIN_FILES} ${SRC_FILES})
  endif(${CUDA})
endif(${EXEF})



