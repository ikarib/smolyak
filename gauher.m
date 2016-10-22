function [ x, w ] = gauher( n, Sigma )
%GAUHER Gauss-Hermite Quadrature Abscissas and Weights
% Given n, this routine returns arrays x[1..n] and w[1..n] containing
% the abscissas and weights of the n-point Gauss-Hermite quadrature formula.
% The largest abscissa is returned in x[1], the most negative in x[n].
% If optional matrix Sigma is given, then abscissas and weights are rescaled
% assuming x are normally distributed with zero mean and covariance matrix Sigma.

m=floor((n+1)/2);
x=nan(1,n); w=nan(1,n);
% The roots are symmetric about the origin, so we have to find only half of them.
for i=1:m % Loop over the desired roots.
    switch i
        case 1  % Initial guess for the largest root.
            z=sqrt(2*n+1)-1.85575*(2*n+1)^-0.16667;
        case 2  % Initial guess for the second largest root.
            z=z-1.14*n^0.426/z;
        case 3  % Initial guess for the third largest root.
            z=1.86*z-0.86*x(1);
        case 4  % Initial guess for the fourth largest root.
            z=1.91*z-0.91*x(2);
        otherwise  % Initial guess for the other roots.
            z=2*z-x(i-2);
    end
    for its=1:10  % Refinement by Newton's method.
        p1=pi^-.25;
        p2=0;
        % Loop up the recurrence relation to get the Hermite polynomial evaluated at z.
        for j=1:n
            p3=p2;
            p2=p1;
            p1=z*sqrt(2/j)*p2-sqrt((j-1)/j)*p3;
        end
        % p1 is now the desired Hermite polynomial. We next compute pp, its derivative, by
        % the relation (4.5.21) using p2, the polynomial of one lower order.
        pp=sqrt(2*n)*p2;
        z1=z;
        z=z1-p1/pp; % Newton's formula.
        if abs(z-z1) <= 3e-14; break; end
    end
    if its==10; error('too many iterations in gauher'); end
    x(i)=z;        % Store the root
    x(n+1-i)=-z;   % and its symmetric counterpart.
    w(i)=2/pp^2;   % Compute the weight
    w(n+1-i)=w(i); % and its symmetric counterpart.
end
if nargin>1
    d=size(Sigma,1); nd=n^d;
    X=nan(nd,d);
    W=nan(nd,d);
    for i=1:d
        ni=n^i;
        X(:,i)=reshape(repmat(x,ni/n,nd/ni),nd,1);
        W(:,i)=reshape(repmat(w,ni/n,nd/ni),nd,1);
    end
    x=X*(sqrt(2)*chol(Sigma));
    w=pi^(-d/2)*prod(W,2);
end