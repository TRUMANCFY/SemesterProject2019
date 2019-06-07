function [v,res] = postprocess(U,V)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
res = U * V';

[m, n] = size(res);

for ind = 1:m
    res(ind,:) = exp(res(ind,:)) / sum(exp(res(ind,:)));
end

v = zeros(m,1);
for ind = 1:m
    [~, v(ind)] = max(res(ind,:));
end
end

