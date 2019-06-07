close all;
clear all;

% load the path first
addpath('./solvers/');
addpath('tensor_toolbox/');
addpath('tensor_toolbox/met/');


load('tensor_we.mat');
t = tensor(t_new);

size(t)

nIter = 50;
% [T1,T2,T3, hist_sgd] = SGDTD(t,nIter,5e-2);
% plot(hist_sgd);

% [T1_l1, T2_l1, T3_l1, hist_sgdl1] = SGDTD_l1(t, nIter, 1e-3, 1e-2, 1e-2);
% plot(hist_sgdl1)

% [T1_l2, T2_l2, T3_l2, hist_sgdl2] = SGDTD_l2(t, nIter, 1e-3, 1e-2);
% plot(hist_sdgl2);

[T1_2nd, T2_2nd, T3_2nd, hist_2nd] = SGDTD_2nd(t, nIter, 0.1, 1e-2);
plot(hist_2nd);
[a,b] = postprocess(T1_2nd,T2_2nd);
% [T1_admm, T2_admm, T3_admm, hist_admm, hist_con] = ADMM_order(t, nIter, 0.1, 1e-2, 1e-1);
% plot(hist_admm);
% hold on
% plot(hist_con);
% legend('ADMM Loss', 'Constrain')
% title('ADMM loss and constrain loss in 100 iterations')








