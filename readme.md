# MWE using cmake to compile MEX and/or binary with or without CUDA support and additional libraries
This work is build upon the following two helpful links:
[CUDA-MEX-CMAKE](https://de.mathworks.com/matlabcentral/fileexchange/45505-cuda-mex-cmake) and [MEX-CMAKE](https://de.mathworks.com/matlabcentral/fileexchange/45522-mex-cmake)

This is a cmake code template to demonstrate how to compile a Matlab MEX file with or without CUDA capability AND additionally how to compile a binary at the same time without MEX support with or without CUDA capability. Furthermore, libraries can be included that 
* are used by the binary only
* are used by cuda only
* are used by the binary, cuda and the mex file.

## Requirements
* [CMake](https://cmake.org/) (mandatory)
* [CUDA](https://developer.nvidia.com/cuda-downloads) (optional)
* [MATLAB](https://de.mathworks.com/) (optional)
* [OpenCV](https://opencv.org/) (mandatory dummy)
* [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) (mandatory dummy)

## Tested
This code has been tested under 
* Ubuntu 16.04; cmake3.5.1; cuda8.0; MATLAB R2016b
* TODO: Test using MacOS and Windows (To compile the test MEX under Windows, first set `MATLAB_ROOT` environment variable to your installed matlab path, then, use cmake or cmake-gui to generate building project according to installed compiler (e.g. MSVS), then, build the generated project using this compiler.)

## Getting started
* Set `MATLAB_ROOT` environment variable in `~/.bashrc` to your installed matlab path, such as
   export `MATLAB_ROOT=/usr/local/MATLAB/R2016b`
* In Terminal do  
     `cd path/to/mweCmakeMexCppCuda/build`  
     `cmake ..`  
     `make`
* in `build/lib` is the mex file and in `build/bin` is the binary
* `cd  ../example`
* execute the matlab script and/or the bash script

## Information
* The cmake file is supposed to be generic enough to easily integrate this to existing code and/or to easily built up on it to generate a versatile cpp/mex/cuda project.

### Possibilities
* Decide whether to generate a mex file: `SET(MEXF {TRUE,FALSE})`
* Decide whether to generate a binary file: `SET(EXEF {TRUE,FALSE})`
* Decide whether to have CUDA support: `SET(CUDA {TRUE,FALSE})`

### File linking
There are multiple integrated ways to link source code e.g.,
* `CPP_MAIN_FILES` are the main files for the binary
* `MEX_MAIN_FILES` are the main files for the mex support
* `COMMON_SRC_FILES` are files that are needed for both, cuda and non-cuda support (supposed to be some middlesman files to easily split cuda from non-cuda support
* `CU_SRC_FILES` are all the cuda files that replace the corresponding cpp files
* `CPP_SRC_FILES` are all the cpp files that replace the cuda files

### Library setup
Additionally, this framework uses OpenCV and Eigen as some dummy libraries to show the versatility of this framework.
* OpenCV is used only for the binary build
* Eigen is used in the case of building MEX and binary files with or without CUDA support
* CUBLAS and CUFFT libraries are linked and can be easily extended

## Tips & Troubleshooting
* It can happen that the building process during `make` fails. This can be due to the fact that the `gpuadd.o` is build twice (haven't found a way to fix this yet). Run `make` a couple of times to overcome this issue or first build with `SET(MEXF TRUE)` and afterwards with `SET(EXEF TRUE)`. For me this error happend every ~10th building process, so I didnt' see the urge of fixing it.
* If Matlab throws an error like
```
Invalid MEX-file '/mweCmakeMexCppCuda/build/lib/mweAddMEX.mexa64':
/MATLAB/R2016b/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /mweCmakeMexCppCuda/build/lib/mweAddMEX.mexa64)
```
then start Matlab from terminal and preload the the `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` library:
```
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6" matlab
```

