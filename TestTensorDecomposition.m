% Tensor Decomposition Test Code
% 06.05.2019 Fengyu Cai

close all;
clear all;
% clc;

% load the path first
addpath('./solvers/');
addpath('tensor_toolbox/');
addpath('tensor_toolbox/met/');

nIter = 200;
% generate the random tensor
T = tenrand([10, 4, 3]);

U = rand(10,3);
V = rand(4,3);
W = rand(3,3);


[T1,T2,T3, hist_sgd] = SGDTD(T,nIter,1e-2,U,V,W);
[T1_als, T2_als, T3_als, hist_als] = SGDTD_2nd(T,nIter,0.1,1e-1,U,V,W);
ref = cp_als(T,3);
fprintf('U difference is %f \n', norm(ref.U{1} - T1_als));
fprintf('V difference is %f \n', norm(ref.U{2} - T2_als));
fprintf('W difference is %f \n', norm(ref.U{3} - T3_als));

fprintf('toolbox reconstruct diff is %f \n', norm(T - reconstruct(ref.U{1}, ref.U{2}, ref.U{3})));




% [T1_reg, T2_reg, T3_reg, hist_reg] = SGDTD_reg(T,nIter,1e-2,1e-1,U,V,W);
% 
% 
% 
% [T1_lalm, T2_lalm, T3_lalm, hist_lalm] = LALM(T,nIter,0.1,1e-2,1e-2,U,V,W);

% use the cp_als to 
% figure(1);
% plot(hist_sgd);
% hold on;
% plot(hist_reg);
% % plot(hist_als);
% % plot(hist_lalm);

% legend('Naive SGD', 'Regularized SGD', 'ALS SGD', 'hist_lalm');

















