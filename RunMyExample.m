clear; clc; close all
% This is an example for a Smolyak functional approximation

% Suppose we want to approximate the following 2-D function 
% The Rosenbrock banana function:
z    = @(x,y)  100*(y-x.^2).^2 + (1-x).^2;
dzdx = @(x,y) -400*(y-x.^2).*x - 2*(1-x);
dzdy = @(x,y)  200*(y-x.^2);

% set approximation levels (in first and second dimensions):
mu=[2 1];

% Try some other examples:
% y = @(x) 2*x(:,1).*exp(-4*x(:,1).^2-16*x(:,2).^2); xm=0; xs=1; mu=[7 7];
% y = @(x) x(:,1)./exp(x(:,2)); xm=[2.5 0]; xs=[2.5 1]; mu=[3 4];

% First we need to construct Smolyak indices S for given mu
% and evaluate Chebyshev polynomial basis function B at extrema grid points X
S = smolyak(mu);

% Construct the test grid scaled according to xm and xs 
% at which we will approximate our true function y(x)
[x,y] = meshgrid(-1:.05:1, -1:.05:1);
z_true=z(x,y);
dzdx_true=dzdx(x,y);
dzdy_true=dzdy(x,y);

% Plot the true function
subplot(2,3,1), surf(x,y,z_true),title('True function, z(x,y)')
subplot(2,3,2), surf(x,y,dzdx_true),title('True derivative, dz/dx')
subplot(2,3,3), surf(x,y,dzdy_true),title('True derivative, dz/dy')

% Estimate coefficients c
S.Values = z(S.GridVectors(:,1),S.GridVectors(:,2));

% Evaluate Smolyak polynomial at test grid x using estimated coefficients c
xy=[x(:) y(:)];
z_fit = S(xy);
dzdx_fit = S(xy,1);
dzdy_fit = S(xy,2);

% Calculate and plot approximation errors
z_err = reshape(z_fit,size(z_true))-z_true;
dzdx_err = reshape(dzdx_fit,size(dzdx_true))-dzdx_true;
dzdy_err = reshape(dzdy_fit,size(dzdy_true))-dzdy_true;
subplot(2,3,4), surf(x,y,z_err),title('Errors in Smolyak interpolation of z(x,y)')
subplot(2,3,5), surf(x,y,dzdx_err),title('Errors in Smolyak interpolation of dz/dx')
subplot(2,3,6), surf(x,y,dzdy_err),title('Errors in Smolyak interpolation of dz/dy')
