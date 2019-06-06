function [Un, Vn, Wn, hist, time_hist] = SGDTD(X, nIter, alpha, U, V, W)
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
%         r = r * r;
        U = rand(m, r) * 1e-1;
        V = rand(n, r) * 1e-1;
        W = rand(k, r) * 1e-1;
    end
    
    norm_tensor = norm(X);
    fprintf('The initial loss before sgd is %f \n', norm(X - reconstruct(U,V,W)) / norm_tensor);
    
    % initialize the history vector
    hist = zeros(nIter+1, 1);
    time_hist = zeros(nIter+1, 1);
    hist(1) = norm(X - reconstruct(U,V,W))/norm_tensor;
    tic;
    for ind=1:nIter
        grad_U = cpGradient(X,U,V,W,1);
        U = U - alpha * grad_U;
        
        grad_V = cpGradient(X,V,U,W,2);
        V = V - alpha * grad_V;
        
        grad_W = cpGradient(X,W,U,V,3);
        W = W - alpha * grad_W;
        hist(ind+1) = norm(X - reconstruct(U,V,W)) / norm_tensor;
        time_hist(ind+1) = toc;
        fprintf('Iteration %d \n', ind);
        fprintf('The loss %f \n', hist(ind+1));
    end
    
    hist(nIter+1) = norm(X - reconstruct(U, V, W)) / norm_tensor;
    
    fprintf('The reconstruct loss after sgd is %f \n', hist(nIter+1));
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
            if mode==1
                Xn(i, r) = V(:,r)' * double(X(i,:,:)) * W(:,r);
            elseif mode==2
                Xn(i, r) = V(:,r)' * double(X(:,i,:)) * W(:,r);
            else
                Xn(i, r) = V(:,r)' * double(X(:,:,i)) * W(:,r);
            end
%             for j = 1:J
%                 for k = 1:K
%                     if mode==1
%                         Xn(i,r) = Xn(i,r) + X(i,j,k) * V(j,r) * W(k,r);
%                     elseif mode==2
%                         Xn(i,r) = Xn(i,r) + X(j,i,k) * V(j,r) * W(k,r);
%                     else
%                         Xn(i,r) = Xn(i,r) + X(j,k,i) * V(j,r) * W(k,r);
%                     end
%                 end
%             end
        end
    end
end

function [gamma] = bigGamma(V, W)
    gamma = (V'*V).*(W'*W);
end



















