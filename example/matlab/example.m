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

tic
ApB_mex = MumfordShahMEX(A,B);
toc

tic
ApB_matlab = A+B;
toc

fprintf('Unit test passed: %d\n', isequal(ApB_mex,ApB_matlab));
fprintf('|ApB_matlab - ApB_mex| =  %f\n', sum(sum(abs(ApB_mex-ApB_matlab))));