function [Un, Vn, Wn, hist] = SGDTD(X, nIter, alpha, U, V, W)
% Naive Tensor Decomposition Methods
% min 1/2|| T - T1 outer* T2 outer* T3 ||^2, given T
    m = size(X,1);
    n = size(X,2);
    k = size(X,3);

    if nargin == 1
        r = min([m n k]);
        nIter = 100;
        alpha = 1e-2;
        U = rand(m, r);
        V = rand(n, r);
        W = rand(k, r);
    elseif nargin == 3
        r = min([m n k]);
        U = rand(m, r);
        V = rand(n, r);
        W = rand(k, r);
    end

    fprintf('The initial loss before sgd is %f \n', norm(X - reconstruct(U,V,W)));
    
    % initialize the history vector
    hist = zeros(nIter+1, 1);
    for ind=1:nIter
        hist(ind) = norm(X - reconstruct(U,V,W));
        grad_U = cpGradient(X,U,V,W,1);
        grad_V = cpGradient(X,V,U,W,2);
        grad_W = cpGradient(X,W,U,V,3);

        U = U - alpha * grad_U;
        V = V - alpha * grad_V;
        W = W - alpha * grad_W;
        
    end
    
    hist(nIter+1) = norm(X - reconstruct(U, V, W));
    
    fprintf('The reconstruct loss after sgd is %f \n', norm(X - reconstruct(U,V,W)));
    Un = U;
    Vn = V;
    Wn = W;
    
end


function [grad] = cpGradient(X,U,V,W,mode)
    grad = -bigX(X,U,V,W,mode) + U * bigGamma(V,W);
end

function [Xn] = bigX(X,U,V,W,mode)
    I = size(U,1);
    R = size(U,2);
    J = size(V,1);
    K = size(W,1);
    
    Xn = zeros(I, R);
    for i = 1:I
        for r = 1:R
            for j = 1:J
                for k = 1:K
                    if mode==1
                        Xn(i,r) = Xn(i,r) + X(i,j,k) * V(j,r) * W(k,r);
                    elseif mode==2
                        Xn(i,r) = Xn(i,r) + X(j,i,k) * V(j,r) * W(k,r);
                    else
                        Xn(i,r) = Xn(i,r) + X(j,k,i) * V(j,r) * W(k,r);
                    end
                end
            end
        end
    end
end

function [gamma] = bigGamma(V, W)
    gamma = (V'*V).*(W'*W);
end



















