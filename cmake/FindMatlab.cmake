# - this module looks for Matlab
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h
#  MATLAB_LIBRARIES:   required libraries: libmex, libmx
#  MATLAB_MEX_LIBRARY: path to libmex
#  MATLAB_MX_LIBRARY:  path to libmx

SET(MATLAB_FOUND 0)
IF( "$ENV{MATLAB_ROOT}" STREQUAL "" )
    MESSAGE(SEND_ERROR "MATLAB_ROOT environment variable not set or found, please do:  
                        export MATLAB_ROOT=/usr/local/MATLAB/R2016b in ~/.bashrc (Linux)  
                        export MATLAB_ROOT=/Applications/MATLAB_R2016b.app in ~/.bash_profile (MacOS)  
                        Add system variable MATLAB_ROOT=D:\\Program Files\\MATLAB\\R2016b (Windows) (not yet tested)" )
ELSE("$ENV{MATLAB_ROOT}" STREQUAL "" )

        FIND_PATH(MATLAB_INCLUDE_DIR mex.h
                  $ENV{MATLAB_ROOT}/extern/include)

        INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})

        FIND_LIBRARY( MATLAB_MEX_LIBRARY
                      NAMES libmex mex
                      PATHS $ENV{MATLAB_ROOT}/bin $ENV{MATLAB_ROOT}/extern/lib 
                      PATH_SUFFIXES
                      glnxa64 glnx86 #linux
                      win64/microsoft win32/microsoft #windows
                      maci maci64) #macos

        FIND_LIBRARY( MATLAB_MX_LIBRARY
                      NAMES libmx mx
                      PATHS $ENV{MATLAB_ROOT}/bin $ENV{MATLAB_ROOT}/extern/lib 
                      PATH_SUFFIXES
                      glnxa64 glnx86 #linux
                      win64/microsoft win32/microsoft #windows
                      maci maci64) #macos

    MESSAGE (STATUS "MATLAB_ROOT: $ENV{MATLAB_ROOT}")

ENDIF("$ENV{MATLAB_ROOT}" STREQUAL "" )

# This is common to UNIX and Win32:
SET(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
)

IF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  SET(MATLAB_FOUND 1)
  MESSAGE(STATUS "Matlab libraries were found.")
ENDIF(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

MARK_AS_ADVANCED(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
)
