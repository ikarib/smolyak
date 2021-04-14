classdef smolyak
% SMOLYAK Computes Smolyak anisotropic grid and basis matrices for subset of Chebyshev polynomials
% USAGE
%   S = smolyak(mu,xm,xs);
%   X = S.GridVectors;
%   S.Values = Y;
%   Yv = S(Xv);
%
% INPUTS
%   mu : the level of approximation in each dimension - vector of length D (number of dimensions)
%   xm : the mean of left and right endpoints = (xmin+xmax)/2 - scalar or vector of length D
%   xs : the spread = (xmax-xmin)/2 - scalar or vector of length D
%
% OUTPUTS
%   S  : matrix of Smolyak indices (previously constructed by calling smolyak with 3 inputs above)
%   X  : D-by-L matrix of the arbitrary evaluation points (default: extrema of order 2^max(mu)+1 Chebyshev polynomial)
%   B  : M-by-L matrix of the polynomial basis functions evaluated at grid points X
%   X  : evaluation points (useful if defaults values are computed) (matrix of Smolyak grid points corresponding to S)
%   S  : max(mu)-by-M matrix of anisotropic Smolyak indices corresponding to the given mu which
%        were selected as a subset of isotropic indices of multidimesional
%        Smolyak elements (grid points and polynomial basis functions)
%        that satisfy the usual isotropic Smolyak rule for the approximation
%        level equal to max(mu)
% Example:
% z    = @(x,y)  100*(y-x.^2).^2 + (1-x).^2;
% dzdx = @(x,y) -400*(y-x.^2).*x - 2*(1-x);
% dzdy = @(x,y)  200*(y-x.^2);
% S = smolyak([2 1]);
% S.Values = z(S.GridVectors(:,1),S.GridVectors(:,2));
% [x,y] = meshgrid(-1:.01:1, -1:.05:1);
% xy=[x(:) y(:)];
% z_err = reshape(S(xy),size(x))-z(x,y);
% dzdx_err = reshape(S(xy,1),size(x))-dzdx(x,y);
% dzdy_err = reshape(S(xy,2),size(x))-dzdy(x,y);
% surf(x,y,z_err)
% surf(x,y,dzdx_err)
% surf(x,y,dzdy_err)

% Copyright (c) 2021, Iskander Karibzhanov (kais@bankofcanada.ca)

    properties
        mu {mustBeNumeric};
        xm {mustBeNumeric};
        xs {mustBeNumeric};
        BasisMatrixInv {mustBeNumeric};
        SmolyakIndices {mustBeNumeric};
        GridVectors {mustBeNumeric};
        Values {mustBeNumeric};
        Coefficients {mustBeNumeric};
    end
    methods
        %constructor
        function self = smolyak(mu,xm,xs)
            D = length(mu); % number of dimensions
            if nargin<2
                xm = zeros(D,1);
            end
            if nargin<3
                xs = ones(D,1);
            end
            mu_max=max(mu);
            smax=2^mu_max;
            
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

            % evaluate at standard nodes
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
            T = cos(pi*(k*(1:smax)));
            T(abs(T)<1e-10)=0;

            % compute Smolyak grid X and basis functions T
            if D>1
                X = xm+xs.*x(S);
            else
                X = xm+xs.*x(S)';
            end
            L = size(S,2);
            T = permute(reshape(T(S,:),D,L,smax),[3 1 2]);
            j=sum(S>1,1); ss=ones(mu_max,L);
            for i=1:mu_max
                ji=(j==i);
                s=S(:,ji);
                sm=s+smax*(0:D-1)';
                ss(1:i,ji)=reshape(sm(s>1),i,size(s,2));
            end
            S=ss;
            
            T = [ones(1,L); reshape(T,smax*D,L)];

            B=T(S(1,:),:);
            for i=2:mu_max
                B=B.*T(S(i,:),:);
            end
            
            self.mu=mu;
            self.xm=xm;
            self.xs=xs;
            self.BasisMatrixInv = inv(B);
            self.GridVectors = X;
            self.SmolyakIndices = S;
        end
        function obj = subsasgn(obj,S,y)
            if S(1).type == '.' && strcmp(S.subs,'Values')
                obj.Values = y;
                obj.Coefficients = obj.Values*obj.BasisMatrixInv;
            else
                error('unknown subscripted assignment')
            end
        end
        % overloaded function
        function y = subsref(obj,S)
            switch S(1).type
                case '.'
                    y = builtin('subsref', obj, S); 
                case '{}'
                  error('{} indexing not supported');
                case '()'
                    X = S.subs{1};
                    D = length(obj.mu); % number of dimensions
                    mu_max=max(obj.mu);
                    smax=2^mu_max;

                    x=(X-obj.xm)./obj.xs;
                    L=size(x,2);
                    T=ones(smax,D,L);
                    T(1,:,:)=x;
                    T(2,:,:)=2*(x.*x)-1;
                    x=permute(2*x,[3 1 2]);
                    for j=3:smax
                        T(j,:,:)=x.*T(j-1,:,:)-T(j-2,:,:);
                    end
                    B=ones(size(obj.SmolyakIndices,2),L);
                    if numel(S.subs)>1 % derivative requested
                        U=ones(smax,D,L);
                        U(2,:,:)=x;
                        for j=3:smax
                            U(j,:,:)=x.*U(j-1,:,:)-U(j-2,:,:);
                        end
                        j = S.subs{2};
                        T(:,j,:) = U(:,j,:).*(1:smax)'; % dT/dxj
                        B(~any(floor((obj.SmolyakIndices+2)/smax)==j),:)=0;
                    end
                    
                    T = [ones(1,L); reshape(T,smax*D,L)];
                    for i=1:mu_max
                        B=B.*T(obj.SmolyakIndices(i,:),:);
                    end
                    y = obj.Coefficients*B;
            end
        end
   end
end
