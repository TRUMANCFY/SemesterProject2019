function [Recon] = reconstruct(U, V, W)
% reconstruct tensor based on three matrices
    [I, R] = size(U);
    J = size(V,1);
    K = size(W,1);

    Recon = zeros(I,J,K);


    for i=1:I
        for j=1:J
            for k=1:K
                Recon(i,j,k) = sum(U(i,:) .* V(j,:) .* W(k,:));
            end
        end
    end
end

