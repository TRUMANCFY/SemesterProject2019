% Tensor Decomposition Test Code
% 06.05.2019 Fengyu Cai

close all;
clear all;
% clc;

% load the path first
addpath('./solvers/');
addpath('tensor_toolbox/');
addpath('tensor_toolbox/met/');

nIter = 100;
% generate the random tensor
% T = tenrand([10, 4, 3]);

U = rand(10,3);
V = rand(4,3);
W = rand(3,3);

T = tensor(reconstruct(U, V, W));

U = rand(10,3);
V = rand(4,3);
W = rand(3,3);

[T1,T2,T3, hist_sgd] = SGDTD(T,nIter,1e-2,U,V,W);
[T1_l1,T2_l1,T3_l1, hist_l1] = SGDTD_l1(T,nIter,1e-2,1e-1,1e-2,U,V,W);
[T1_l2, T2_l2, T3_l2, hist_l2] = SGDTD_l2(T,nIter,1e-2,1e-1,U,V,W);
[T1_2nd, T2_2nd, T3_2nd, hist_2nd] = SGDTD_2nd(T,nIter,1e-2,1e-1,U,V,W);
[T1_lalm, T2_lalm, T3_lalm, hist_admm] = ADMM(T,nIter,1e-2,1e-2,1e-2,U,V,W);

figure();
plot(hist_sgd);
hold on;
plot(hist_l1);
plot(hist_l2);
plot(hist_2nd);
plot(hist_admm);

legend('Naive SGD', 'L1 SGD', 'L2 SGD', 'Newton Method', 'ADMM');




