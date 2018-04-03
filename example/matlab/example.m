%%
clc
clear all;

addpath('../../build/lib/')

%%
max_limit = 1;
dim_r = randi(max_limit);
dim_c = randi(max_limit);

A = rand(dim_r, dim_c);
B = rand(dim_r, dim_c);

if exist('MumfordShahCUDA.mexa64','file')
fprintf('Invoke CUDA version\n');
tic
ApB_mex = MumfordShahCUDA(A,B);
toc
elseif exist('MumfordShah.mexa64','file')
fprintf('Invoke CUDA version\n');
tic;
ApB_mex = MumfordShah(A,B);
toc
end

tic
ApB_matlab = A+B;
toc

fprintf('Unit test passed: %d\n', isequal(ApB_mex,ApB_matlab));
fprintf('|ApB_matlab - ApB_cuda| =  %f\n', sum(sum(abs(ApB_mex-ApB_matlab))));