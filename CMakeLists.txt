cmake_minimum_required(VERSION 2.8)
project(MumfordShah)

#set output of binary and library (mex) files
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # add FindMatlab module

add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
add_definitions(/DMX_COMPAT_32)

find_package(Matlab)
if(MATLAB_FOUND)
  add_definitions(-DUSE_MEX)
endif(MATLAB_FOUND)

#find_package(CUDA)
if(CUDA_FOUND)
  add_definitions(-DUSE_CUDA)
endif(CUDA_FOUND)

#find_package(OpenCV)
if(OPENCV_FOUND)
  add_definitions(-DUSE_OPENCV)
endif(OPENCV_FOUND)


######################################################
##################CUDA & MEX##########################
######################################################
if(MATLAB_FOUND AND CUDA_FOUND)#build mex file with cuda support
  message(STATUS "MATLAB and CUDA found, CUDA MEX will be compiled.")
    
  # compile cuda mex for shader models 2.0 as default, support double
  set(CU_MEX_LIB_NAME "${PROJECT_NAME}CUDA")
  
  # set up cuda & matlab libraries
  set(SRC_FILES src/cudaAdd.cu)
  set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_20,code="sm_20,compute_20" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)
  include_directories(${MATLAB_INCLUDE_DIR})
  cuda_compile(${CU_MEX_LIB_NAME} ${SRC_FILES} SHARED)
  add_library(${CU_MEX_LIB_NAME} SHARED ${${CU_MEX_LIB_NAME}} ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
  target_link_libraries(${CU_MEX_LIB_NAME} ${MATLAB_LIBRARIES} ${CUDA_LIBRARIES})
  set_target_properties(${CU_MEX_LIB_NAME} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
  
  # 32-bit or 64-bit mex
  if(WIN32)
      if (CMAKE_CL_64)
        set_target_properties(${CU_MEX_LIB_NAME} PROPERTIES SUFFIX .mexw64)
      else(CMAKE_CL_64)
        set_target_properties(${CU_MEX_LIB_NAME} PROPERTIES SUFFIX .mexw32)
      endif(CMAKE_CL_64)
  else(WIN32)
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        set_target_properties(${CU_MEX_LIB_NAME} PROPERTIES SUFFIX .mexa64 PREFIX "")
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
        set_target_properties(${CU_MEX_LIB_NAME} PROPERTIES SUFFIX .mexglx PREFIX "")
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
  endif(WIN32)

  # install to /bin by default
#  install(TARGETS ${CU_MEX_LIB_NAME} DESTINATION lib)

#else(MATLAB_FOUND AND CUDA_FOUND)
#    message(STATUS "MATLAB and/or CUDA not found, CUDA MEX will not be compiled.")
endif(MATLAB_FOUND AND CUDA_FOUND)

######################################################
##################MEX ONLY############################
######################################################
if(MATLAB_FOUND AND NOT CUDA_FOUND)#build mex file for c++ without cuda support
  message(STATUS "MATLAB found and CUDA not found, MEX without CUDA support will be compiled.")

# compile mex
  set(CPP_MEX_LIB_NAME "${PROJECT_NAME}")

# set up matlab libraries
set(CPP_FILE_NAME src/cudaAdd.cpp)
INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})
add_library(${CPP_MEX_LIB_NAME} SHARED ${CPP_FILE_NAME} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
target_link_libraries(${CPP_MEX_LIB_NAME} ${MATLAB_LIBRARIES})

# 32-bit or 64-bit mex
if(WIN32)
  if (CMAKE_CL_64)
      SET_TARGET_PROPERTIES(${CPP_MEX_LIB_NAME} PROPERTIES SUFFIX .mexw64)
  else(CMAKE_CL_64)
      SET_TARGET_PROPERTIES(${CPP_MEX_LIB_NAME} PROPERTIES SUFFIX .mexw32)
  endif(CMAKE_CL_64)
else(WIN32)
  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
      SET_TARGET_PROPERTIES(${CPP_MEX_LIB_NAME} PROPERTIES SUFFIX .mexa64 PREFIX "")
  else(CMAKE_SIZEOF_VOID_P MATCHES "8")
      SET_TARGET_PROPERTIES(${CPP_MEX_LIB_NAME} PROPERTIES SUFFIX .mexglx PREFIX "")
  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)

#else(MATLAB_FOUND NOT CUDA_FOUND)
#  message(STATUS "MATLAB not found, MEX will not be compiled.")
endif(MATLAB_FOUND AND NOT CUDA_FOUND)

######################################################
#############CUDA EXECUTABLE##########################
######################################################
if(OPENCV_FOUND AND CUDA_FOUND)#build executable
  message(STATUS "OPENCV and CUDA found, CUDA EXECUTABLE will be compiled.")
#TODO
#else(OPENCV_FOUND AND CUDA_FOUND)
#  message(STATUS "OPENCV and/or CUDA not found, CUDA EXECUTABLE will not be compiled.")
endif(OPENCV_FOUND AND CUDA_FOUND)

######################################################
#############CPP EXECUTABLE ONLY######################
######################################################
if(OPENCV_FOUND AND NOT CUDA_FOUND)#build executable
  message(STATUS "OPENCV found and CUDA not found, CPP EXECUTABLE without CUDA support will be compiled.")
#TODO
#else(OPENCV_FOUND AND NOT CUDA_FOUND)
#  message(STATUS "OPENCV not found, CPP EXECUTABLE will not be compiled.")
endif(OPENCV_FOUND AND NOT CUDA_FOUND)


