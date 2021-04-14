clear; clc
gam     = 1;      % Utility-function parameter
alpha   = 0.36;   % Capital share in output
beta    = 0.99;   % Discount factor
delta   = 0.025;  % Depreciation rate 
rho     = 0.95;   % Persistence of the log of the productivity level
sigma   = sqrt(.0002);   % Standard deviation of shocks to the log of the productivity level
A=(1/beta-1+delta)/alpha;
tic
mu=[3 2]; xm=1; xs=.2;
[B,X,S]=smolyak(mu,xm,xs);
n=size(B,1);
m=10; [e,w]=gauher(m,sigma^2); % Gauss-Hermite quadrature with m nodes
k=X(1,:); a=X(2,:);
% kp=A*alpha*beta*k.^alpha;
% b=B\kp;
b=zeros(1,n); b(1)=1;
Bi=inv(B); bdamp=0.05;
kp=b*B;
ap=exp(e)*a.^rho;
for it=1:1000
    if any(kp<0); error('negative capital'); end
	x=[repmat(kp,1,m); reshape(ap',1,n*m)];
    kpp=reshape(b*smolyak(mu,xm,xs,S,x),n,m)';
    uc=(A*a.*k.^alpha+(1-delta)*k-kp).^-gam;
    ucp=(A*ap.*repmat(kp.^alpha,m,1)+repmat((1-delta)*kp,m,1)-kpp).^-gam;
    r=1-delta+repmat(A*alpha*kp.^(alpha-1),m,1).*ap;
    y=bdamp*beta*(w'*(ucp.*r))./uc+(1-bdamp);
	kp=y.*kp;
	b=kp*Bi;
    dkp=mean(abs(1-y));
    fprintf('iter=%g \t diff=%g\n',it,dkp)
    if dkp<1e-10; break; end
end
time_Smol = toc;
%% compute Euler equation errors
tic
T=10000; T_test=10200; discard=200;
a20200=exp(sigma*randn(T+T_test,1));
% load Smolyak_Anisotropic_JMMV_2014\aT20200N10
x = [ones(1,T_test+1); a20200(T+1:T+T_test,1)' nan];
for t=1:T_test
    x(1,t+1)=b*smolyak(mu,xm,xs,S,x(:,t));
%     x(2,t+1)=x(2,t)^rho*E(t);
end
x=x(:,1+discard:end);
n=T_test-discard;
k=x(1,1:n); kp=x(1,2:n+1);
a=x(2,1:n); ap=exp(e)*a.^rho;
uc=(A*a.*k.^alpha+(1-delta)*k-kp).^-gam;
kpp=reshape(b*smolyak(mu,xm,xs,S,[repmat(kp,1,m); reshape(ap',1,n*m)]),n,m)';
ucp=(A*ap.*repmat(kp.^alpha,m,1)+repmat((1-delta)*kp,m,1)-kpp).^-gam;
r=1-delta+repmat(A*alpha*kp.^(alpha-1),m,1).*ap;
err=1-beta*(w'*(ucp.*r))./uc; % Unit-free Euler-equation errors
err_mean=log10(mean(abs(err)));
err_max=log10(max(abs(err)));
time_test = toc;
%% Display the results
disp(' '); disp('           SMOLYAK OUTPUT:'); disp(' '); 
disp('RUNNING TIME (in seconds):'); disp('');
disp('a) for computing the solution'); 
disp(time_Smol);
disp('b) for implementing the accuracy test'); 
disp(time_test);
disp('APPROXIMATION ERRORS (log10):'); disp(''); 
disp('a) mean error across 4N+1 optimality conditions'); 
disp(err_mean)
disp('b) max error across 4N+1 optimality conditions'); 
disp(err_max)
