addpath('../../build/lib/')

dim_r = randi(10000);
dim_c = randi(10000);

A = rand(dim_r, dim_c);
B = rand(dim_r, dim_c);

tic
% ApB_cuda = MumfordShahCUDA(A,B);
ApB_cuda = MumfordShah(A,B);
toc
tic
ApB_matlab = A+B;
toc

fprintf('Unit test passed: %d\n', isequal(ApB_cuda,ApB_matlab));
fprintf('|ApB_matlab - ApB_cuda| =  %f\n', sum(sum(abs(ApB_cuda-ApB_matlab))));