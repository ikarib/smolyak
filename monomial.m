function [ x, w ] = monomial( N, Sigma, type )
%MONOMIAL Computes Abscissas and Weights for Monomial Integration Rules
%   n - number of dimensions
%   Sigma - covariance matrix
%   type - 1 for M1, 2 for M2
Omega = chol(Sigma);
switch type
    case 1
        R=sqrt(N)*Omega;
        x=kron(R,[1;-1]);
        w=repmat(.5/N,2*N,1);
    case 2
        R=sqrt(2+N)*Omega;
        D=sqrt(1+N/2)*Omega;
        x=[zeros(1,N);kron(R,[1;-1])];
        for i=1:N-1
            for j=i+1:N
                x=[x;[1 1;-1 1;1 -1;-1 -1]*D([i j],:)]; %#ok<AGROW>
            end
        end
        w=[2/(2+N);repmat((4-N)/2/(2+N)^2,2*N,1);repmat(1/(N+2)^2,2*N*(N-1),1)];
end