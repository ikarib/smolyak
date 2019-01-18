function [B,X,S] = smolyak(mu,xm,xs,S,X)
% SMOLYAK Computes Smolyak anisotropic grid and basis matrices for subset of Chebyshev polynomials
% USAGE
%   [B,X,S] = smolyak(mu,xm,xs);
%   B = smolyak(mu,xm,xs,S,X);
%
% INPUTS
%   mu : the level of approximation in each dimension - vector of length D (number of dimensions)
%   xm : the mean of left and right endpoints = (xmin+xmax)/2 - scalar or vector of length D
%   xs : the spread = (xmax-xmin)/2 - scalar or vector of length D
%   S  : matrix of Smolyak indices (previously constructed by calling smolyak with 3 inputs above)
%   X  : D-by-L matrix of the arbitrary evaluation points (default: extrema of order 2^max(mu)+1 Chebyshev polynomial)
%
% OUTPUTS
%   B  : M-by-L matrix of the polynomial basis functions evaluated at grid points X
%   X  : evaluation points (useful if defaults values are computed) (matrix of Smolyak grid points corresponding to S)
%   S  : max(mu)-by-M matrix of anisotropic Smolyak indices corresponding to the given mu which
%        were selected as a subset of isotropic indices of multidimesional
%        Smolyak elements (grid points and polynomial basis functions)
%        that satisfy the usual isotropic Smolyak rule for the approximation
%        level equal to max(mu)

% Copyright (c) 2016, Iskander Karibzhanov (kais@bankofcanada.ca)

if nargin<3, error('at least 3 parameters must be specified'), end
if nargin<4, S=[]; end
if nargin<5, X=[]; end

D = length(mu); % number of dimensions
mu_max=max(mu);
smax=2^mu_max;

if isempty(S)
    if mu_max>0
        S=mat2cell(1:smax+1,1,2.^[0 1 1:mu_max-1]);
    else
        S={1};
    end
    S1=S;
    for i=mu(1)+2:mu_max+1
        S{i}=zeros(1,0);
    end
    for j=2:D
        for i=mu_max+1:-1:2
            s=cell(1,mu_max+1);
            for k=1:min(i,mu(j)+1)
                s{k}=[repmat(S{i+1-k},size(S1{k}));
                      reshape(repmat(S1{k},size(S{i+1-k},2),1),1,[])];
            end
            S{i}=[s{:}];
        end
        S{1}=ones(j,1);
    end
    S=[S{:}];
end

M=size(S,2);
if isempty(X) % evaluate at standard nodes
    if mu_max>0
        k=zeros(smax+1,1);
        k(1:2)=[1/2 1];
        for i=2.^(2:mu_max)
            k(i/2+2:i+1)=(i-1:-2:1)/i;
        end
    else
        k=1/2;
    end
    x = cos(pi*k); % extrema of Chebyshev polynomial
    phi = cos(pi*(k*(1:smax)));
    phi(abs(phi)<1e-10)=0;

    % compute Smolyak grid X and basis functions phi
    X = xm+xs.*x(S);
    phi = permute(reshape(phi(S,:),D,M,smax),[3 1 2]);
    L=M;
    j=sum(S>1); ss=ones(mu_max,M);
    for i=1:mu_max
        ji=(j==i);
        s=S(:,ji);
        sm=s+smax*(0:D-1)';
        ss(1:i,ji)=reshape(sm(s>1),i,size(s,2));
    end
    S=ss;
else % evaluate at arbitrary nodes X
    x=(X-xm)./xs;
    L=size(x,2);
    phi=ones(smax,D,L);
    phi(1,:,:)=x;
    phi(2,:,:)=2*(x.*x)-1;
    x=permute(2*x,[3 1 2]);
    for j=3:smax
        phi(j,:,:)=x.*phi(j-1,:,:)-phi(j-2,:,:);
    end
end
phi = [ones(1,L); reshape(phi,smax*D,L)];

B=phi(S(1,:),:);
for i=2:mu_max
    B=B.*phi(S(i,:),:);
end
