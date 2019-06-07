function [Un, Vn, Wn, hist, time_hist] = SGDTD_2nd(X, nIter, alpha, rho, U, V, W)
% regularized SGD Tensor Decomposition Methods
% min 1/2|| T - T1 outer* T2 outer* T3 ||^2, given T
    m = size(X,1);
    n = size(X,2);
    k = size(X,3);

    if nargin == 1
        r = min([m n k]);
        nIter = 100;
        alpha = 1e-2;
        rho = 0.1;
        U = rand(m, r);
        V = rand(n, r);
        W = rand(k, r);
    elseif nargin == 4
        r = min([m n k]);
        r = 50;
        U = rand(m, r) * 1e-1;
        V = rand(n, r) * 1e-1;
        W = rand(k, r) * 1e-1;
    end
    norm_tensor = norm(X);
    
    hist = zeros(nIter+1,1);
    hist(1) = norm(X - reconstruct(U,V,W))/ norm_tensor;
    fprintf('The initial loss before regularization sgd is %f \n', hist(1));
    
    time_hist = zeros(nIter, 1);
    
    
    tic;
    for ind=1:nIter
        U_new = (1 - alpha) * U + alpha * star(X,U,V,W,rho,1);
        U_new(U_new < 0) = 0;
%         size(U_new)
        U_new = proximalOp_nuclear(U_new, 0.01);
        
        V_new = (1 - alpha) * V + alpha * star(X,V,U_new,W,rho,2);
        V_new(V_new < 0) = 0;
        V_new = proximalOp_nuclear(V_new, 0.01);
        
        W_new = (1 - alpha) * W + alpha * star(X,W,U_new,V_new,rho,3);
        W_new(W_new < 0) = 0;
        W_new = proximalOp_nuclear(W_new, 0.01);
        
        U = U_new;
        V = V_new;
        W = W_new;
        
        hist(ind+1) = norm(X - reconstruct(U,V,W)) / norm_tensor;
        fprintf('Iteration: %d \n', ind);
        fprintf('The loss %f \n', hist(ind+1));
        time_hist(ind+1) = toc;
        
        if hist(ind+1) > hist(ind)
            break
        end
    end
    
    hist(nIter+1) = norm(X - reconstruct(U,V,W)) / norm_tensor;
    
    fprintf('The reconstruct loss after regularization sgd is %f \n', hist(ind+1));
    Un = U;
    Vn = V;
    Wn = W;
    
end


function [grad] = star(X,U,V,W,rho,mode)
    grad = bigX(X,U,V,W,mode) / bigGamma(V,W,rho);
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

function [gamma] = bigGamma(V, W, rho)
    r = size(V,2);
    gamma = (V'*V).*(W'*W) + rho * eye(r);
end


