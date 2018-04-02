Useful links:
https://de.mathworks.com/matlabcentral/fileexchange/45505-cuda-mex-cmake
https://de.mathworks.com/matlabcentral/fileexchange/45522-mex-cmake




This is a MEX code template with cmake build tree to demonstrate how to compile a Matlab MEX 
file with CUDA GPU support using cmake. cmake and CUDA are needed for compiling the MEX.

To compile the test MEX under Linux, 
first set MATLAB_ROOT environment variable to your installed matlab path,
such as 'export MATLAB_ROOT=/usr/local/MATLAB/R2012b',
then, simply do

mkdir build
cd build
cmake ../src
make
make install

To compile the test MEX under Windows,
first set MATLAB_ROOT environment variable to your installed matlab path,
then, use cmake or cmake-gui to generate building project according to installed compiler (e.g. MSVS),
then, build the generated project using this compiler.

The test MEX source code is located under /src/cudamex/cudaAdd. The compiled test MEX 'cudaAdd' will 
be installed into /bin by default. C=cudaAdd(A,B) basically do element-by-element addition for 1D or 
2D matrix A and B, return matrix C. The code is not optimized for speed purpose, but for demonstrating 
basic CUDA MEX code structure.

To add new MEX source code, for example cudaXXX.cu, simply do
1. add a new folder 'cudaXXX' under /src/cudamex
2. add one line 'add_subdirectory(cudaXXX)' to CMakeLists.txt under /src/cudamex
3. copy CMakeLists.txt under /src/cudamex/cudaAdd to folder /src/cudamex/cudaXXX
4. change first line to set(CU_FILE cudaXXX) in copied CMakeLists.txt
5. follow compiling steps as described above


