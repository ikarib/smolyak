clear; clc; close all
% This is an example for a Smolyak functional approximation

% Suppose we want to approximate the following 2-D function 
% The Rosenbrock banana function:
y = @(x) 100*(x(:,2)-x(:,1).^2).^2+(1-x(:,1)).^2;

% specify domain: mean xm=(max+min)/2, and spread xs=(max-min)/2
xm=0; xs=1;

% set approximation levels (in first and second dimensions):
mu=[2 1];

% Try some other examples:
% y = @(x) 2*x(:,1).*exp(-4*x(:,1).^2-16*x(:,2).^2); xm=0; xs=1; mu=[7 7];
% y = @(x) x(:,1)./exp(x(:,2)); xm=[2.5 0]; xs=[2.5 1]; mu=[3 4];

% First we need to construct Smolyak indices S for given mu
% and evaluate Chebyshev polynomial basis function B at extrema grid points X
[B,X,S] = smolyak(mu,xm,xs);

% Construct the test grid scaled according to xm and xs 
% at which we will approximate our true function y(x)
[x1,x2] = meshgrid(-1:.05:1, -1:.05:1);
x=bsxfun(@plus,bsxfun(@times,[x1(:) x2(:)],xs),xm);
y_true=y(x);

% Plot the true function
x1=reshape(x(:,1),size(x1));
x2=reshape(x(:,2),size(x2));
y_true = reshape(y_true,size(x1));
subplot(1,2,1), surf(x1,x2,y_true),title('True function')

% Estimate coefficients c
c = B\y(X);

% Evaluate Smolyak polynomial at test grid x using estimated coefficients c
y_fit=smolyak(mu,xm,xs,S,x)*c;

% Calculate and plot approximation errors
y_err = y_fit-y_true(:);
subplot(1,2,2), scatter3(x1(:),x2(:),y_err,'.'),title('Errors in Smolyak interpolation')