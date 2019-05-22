function [X] = proximalOp_nuclear(A, lambda)
% The proximal operator of nuclear norm behaves like:
% prox lambda| |*(A) = argminX 1/2 |X-A|^2 + lambda |X|*
% both A and X are matrix

% do svd first
[U,S,V] = svd(A,0);
diagS = diag(S);
diagS_update = proximalOp_l1(diagS,lambda);
X = U * diag(diagS_update) * V';
end

