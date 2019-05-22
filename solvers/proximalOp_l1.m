function [x] = proximalOp_l1(a,lambda)
% Proximal Operator l1 norm:
% proxlambda = argmin x 1/2|x-a|^2 + lambda * |x|1

% the result should be Prox(a)i = sign(ai) max(|a|-lambda, 0)
x = sign(a) .* max(abs(a)-lambda,0);
end

