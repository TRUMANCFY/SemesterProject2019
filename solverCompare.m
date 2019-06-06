% This file is used to compare the performance of gradient descent

close all;
clear all;

% load the path first
addpath('./solvers/');
addpath('tensor_toolbox/');
addpath('tensor_toolbox/met/');
load('tensor_0_7.mat');

U = rand(10,3);
V = rand(4,3);
W = rand(3,3);

t = tensor(reconstruct(U, V, W));

m = size(t, 1);
n = size(t, 2);
k = size(t, 3);
r = min([m, n, k]);

U = rand(m, r) * 1e-1;
V = rand(n, r) * 1e-1;
W = rand(k, r) * 1e-1;
nIter = 1000;

[T1_sgd, T2_sgd, T3_sgd, hist_sgd, time_sgd] = SGDTD(t,nIter,1e-1,U,V,W);
[T1_2nd, T2_2nd, T3_2nd, hist_2nd, time_2nd] = SGDTD_2nd(t, nIter, 0.1, 1e-2,U,V,W);

figure();
semilogy(hist_sgd);
hold on;
semilogy(hist_2nd, '*');
title('The comparison between 1st order descent and Newton Method');
legend('1st order descent', 'Newton Method');
xlabel('Epochs');
ylabel('Error');

figure();
semilogy(time_sgd, hist_sgd);
hold on;
semilogy(time_2nd, hist_2nd, '*');
title('The comparison between 1st order descent and Newton Method');
legend('1st order descent', 'Newton Method');
xlabel('Time(second)');
ylabel('Error');