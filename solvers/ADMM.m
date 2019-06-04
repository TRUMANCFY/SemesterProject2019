function [Un, Vn, Wn, hist, hist_con] = ADMM(X, nIter, alpha,mu, rho, U, V, W)
% Linearized Augmented Lagrangian Methods for the tensor decomposition
% min 1/2|| T - T1 outer* T2 outer* T3 ||^2, given T
% s.t. U * 1 = 1 (the sum along with the row)

    m = size(X,1);
    n = size(X,2);
    k = size(X,3);

    if nargin == 1
        r = min([m n k]);
        nIter = 100;
        alpha = 1e-1;
        mu = 1e-3;
        rho = 0.01;
        U = rand(m, r);
        V = rand(n, r);
        W = rand(k, r);
    elseif nargin == 5
        r = min([m n k]);
        U = rand(m, r) * 1e-1;
        V = rand(n, r) * 1e-1;
        W = rand(k, r) * 1e-1;
    end
    
    I = size(U,1);
    R = size(U,2);
    
    Y = rand(I,1);
    
    norm_tensor = norm(X);

    fprintf('The initial loss before admm is %f \n', norm(X - reconstruct(U,V,W))/norm_tensor);
    
    fprintf('The contrain loss before admm is %f \n', norm(sum(U,2) - ones(I,1)));
    % initialize the history
    hist = zeros(nIter+1,1);
    hist_con = zeros(nIter+1,1);
    
    for ind = 1:nIter
       hist(ind) = norm(X - reconstruct(U,V,W)) / norm_tensor;
       hist_con(ind) = norm(sum(U,2) - ones(I,1));
       fprintf('Iter %d \n', ind);
       fprintf('The loss of admm is %f \n', hist(ind));
       fprintf('The loss of constrain is %f \n', hist_con(ind));
       U_new = proximalOp_nuclear(update_U(X,U,V,W,Y,mu,rho), 0.01);
       U_new(U_new < 0) = 0;

       V_new = (1-alpha) * V + alpha * star(X,V,U_new,W,rho,2);
       W_new = (1-alpha) * W + alpha * star(X,W,U_new,V,rho,3);
%        V_new = update_U(X,)
       
       % update the Lagragian Multiplier
       Y = Y + alpha * (U_new * ones(R,1) - ones(I,1));
       
       U = U_new;
       V = V_new;
       W = W_new;
    end
    hist(nIter + 1) = norm(X - reconstruct(U,V,W)) / norm_tensor;
    fprintf('The initial loss after admm is %f \n', norm(X - reconstruct(U,V,W))/norm_tensor);
    fprintf('The contrain loss after admm is %f \n', norm(sum(U,2) - ones(I,1)));
    
    Un = U;
    Vn = V;
    Wn = W;
    
end

function [grad] = star(X,U,V,W,rho,mode)
    r = size(V, 2);
    grad = bigX(X,U,V,W,mode) / (bigGamma(V,W,rho) + 0.01*eye(r));
end

function [U_update] = update_U(X,U,V,W,Y,mu,rho)
    [I,R] = size(U);

    U_update = (bigX(X,U,V,W,1) + mu * ones(I,1) * ones(1,R) - Y * ones(1,R)) / (bigGamma(V,W,rho) + mu*ones(R,1)*ones(1,R) + 0.01*eye(R));
    %     U_update = (bigX(X,U,V,W,1) + mu*U - Y - mu*(sum(U,2)-ones(I,1)) * ones(1,R)) / (mu*eye(R) + bigGamma(V,W,rho));
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
    
    
    
    
    
    
    


