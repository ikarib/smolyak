% see MATLAB Tutorial on how to run CUDA or PTX Code on GPU:
% http://www.mathworks.com/help/distcomp/run-cuda-or-ptx-code-on-gpu.html
max_iter=10000; disp_iter=100;
N     = 10;     % Number of countries
gam   = 1;      % Utility-function parameter
alpha = 0.36;   % Capital share in output
beta  = 0.99;   % Discount factor
delta = 0.025;  % Depreciation rate 
rho   = 0.95;   % Persistence of the log of the productivity level
Sigma = 0.01^2*(eye(N)+ones(N)); % Covariance matrix of shocks to the log of the productivity level
A=(1/beta-1+delta)/alpha; D=2*N;
mu = repmat(3,1,D); % States: k[1..N], a[1..N]
if any(size(mu)~=[1,D]); error('mu must be a row vector of length %d',D); end
gssa=false; % use initial guess from GSSA
if gssa
    addpath('Smolyak_Anisotropic_JMMV_2014')
    load(sprintf('gssa_N=%d.mat',N));
    xm=(min_simul+max_simul)'/2; xs=(max_simul-min_simul)'/2;
    clear min_simul max_simul
else
    xm=ones(D,1); xs=repmat(0.2,D,1); % set mean xm to steady state and spread xs to 20% away
end
[B,X,S]=smolyak(mu,xm,xs);
if gssa % to check with JMMV reorder rows using j indices
    [~,i]=sortrows(Smolyak_Elem_Anisotrop(Smolyak_Elem_Isotrop(D,max(mu)),mu));
    [~,j]=sortrows(S); [~,j]=sort(j); i=i(j); clear j
end

J=11;
if J>10
    [e,w]=monomial(N,Sigma,J-10); % M1 or M2 monomial rule
else
    [e,w]=gauher(J,Sigma); % Gauss-Hermite quadrature with J nodes
end
e=permute(exp(e),[2 3 1]);
w=permute(w,[2 3 1]);

J=size(e,3);
M=size(B,2);

k=X(1:N,:); a=X(N+1:2*N,:); clear X
ap=e.*a.^rho;

try
    gpu = gpuDeviceCount;
catch
    gpu = 0;
end
% if isunix % cluster
%     [~,Cfg]=system('scontrol show node n28 | grep CfgTRES= | tr "," "\n" | grep gres/gpu | cut -d = -f 2');
%     [~,Alloc]=system('scontrol show node n28 | grep AllocTRES= | tr "," "\n" | grep gres/gpu | cut -d = -f 2');
%     if isempty(Alloc); Alloc='0'; end
%     gpu = str2double(Cfg)-str2double(Alloc)
%     setenv('MATLAB_WORKER_ARGS',sprintf('--gres=gpu:%d',gpu))
% end
if gpu
    if gpu>1
        p = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(p)
            distcomp.feature('LocalUseMpiexec',false)
            parpool('local',gpu)
        elseif p.NumWorkers ~= gpu
            delete(p)
            parpool(gpu)
        end
    end
    fname = sprintf('smolyak_N%d_mu%d.ptx',N,max(mu));
    if ~exist(fname,'file')
        if system(sprintf('nvcc -arch=sm_35 -ptx smolyak.cu -DMU=%d -DN=%d -DD=%d -DM=%d -o %s',max(mu),N,D,M,fname)); error('nvcc failed'); end
    end
    x=[nan(N,M,J); ap];
%     spmd
        gd = gpuDevice;
        fprintf('GPU%d: %s\n', gd.Index, gd.Name)
        kernel = parallel.gpu.CUDAKernel(fname, 'smolyak.cu');
        kernel.ThreadBlockSize = 128;
        kernel.GridSize = ceil(M*J/kernel.ThreadBlockSize(1));
        setConstantMemory(kernel,'xm',xm);
        setConstantMemory(kernel,'xs',xs);
        setConstantMemory(kernel,'s',uint8(S-1));
        kpp_=nan(N,M,J/gpu,'gpuArray');
        x=gpuArray(x(:,:,1+J/gpu*(labindex-1):J/gpu*labindex));
%     end
else
    kpp=nan(N,M,J);
end
%% main loop
bfile=sprintf('b_N=%d_mu=%d.mat',N,max(mu));
if exist(bfile,'file')
    fprintf('loading %s\n',bfile)
    load(bfile)
elseif gssa
    c=smolyak(mu,0,1,S,simul_norm')\k_prime_GSSA;
else
    c=zeros(N,M); c(:,1)=1;
end
kp=c*B;
bdamp=0.05;
binvfile=sprintf('B_inv_N=%d_mu=%d.mat',N,max(mu));
if exist(binvfile,'file')
    fprintf('loading %s ... ',binvfile)
    load(binvfile)
    fprintf('done\n')
else
    B_inv=inv(B);
%    save(binvfile,'B_inv')
end
if gpu
%     spmd
        kp_=gpuArray(kp);
        B_inv=gpuArray(B_inv);
        c=gpuArray(c);
        % Measure the overhead introduced by calling the wait function.
        tover = inf;
        for itr = 1:100
            tic;
            wait(gd);
            tover = min(toc, tover);
        end
%     end
end
t0=0; % total runtime
t1=0; % memcpy_in time
t2=0; % kernel time
t3=0; % memcpy_out time
t4=0; % host time
t5=0; % host time
t6=0; % host time
fprintf('Iter\tGFLOPS\tMemcpy_IN, MB/s\tMemcpy_OUT, MB/s\tcollect\tcpu\tgemm\tRuntime\tDiff\n')
%profile on
tic
%%
for it=1:max_iter
    if any(kp(:)<0); error('negative capital'); end
    if it==10; bdamp=0.1; end
    if gpu
%         spmd
tmp=tic;
            x(1:N,:,:)=repmat(kp_,1,1,J/gpu);
t1=t1+toc(tmp);
tmp=tic;
            kpp_ = feval(kernel, x, kpp_, M*J/gpu, c);
            wait(gd);
t2=t2+toc(tmp)-tover;
tmp=tic;
%            kpp(:,:,1+J/gpu*(labindex-1):J/gpu*labindex) = gather(kpp_);
            kpp = gather(kpp_);
            wait(gd);
t3=t3+toc(tmp);
%         end
tmp=tic;
%         t1=t1{1};t2=t2{1};t3=t3{1};
%         kpp=cat(3,kpp{:});
t4=t4+toc(tmp);
    else
tmp=tic;
        for j=1:J
            kpp(:,:,j)=c*smolyak(mu,xm,xs,S,[kp; ap(:,:,j)]);
        end
t2=t2+toc(tmp);
    end
tmp=tic;
    r=1-delta+A*alpha*kp.^(alpha-1).*ap;
    ucp=sum((w.*mean(A*kp.^alpha.*ap-kpp+(1-delta)*kp).^-gam).*r,3);
    uc=mean(A*a.*k.^alpha+(1-delta)*k-kp).^-gam;
    err=1-beta*ucp./uc; % Unit-free Euler-equation errors
    kp=kp.*(1-bdamp*err);
t5=t5+toc(tmp);
tmp=tic;
    if gpu
%         spmd
            kp_ = gpuArray(kp);
            c = kp_*B_inv;
%         end
    else
        c = kp*B_inv;
    end
t6=t6+toc(tmp);
    if ~mod(it,disp_iter)
        dkp=mean(abs(err(:)));
        tmp=t0; t0=toc; tmp=t0-tmp; gflops=M*J*M*(2*N+max(mu)-1)/t2*disp_iter/1e9;
        fprintf('%g\t%.1f (%.1f%%)\t%.1f (%.1f%%)\t%.1f (%.1f%%)\t%.1f%%\t%.1f%%\t%.1f%%\t%.1f%%\t%.1f\t%e\n',it,gflops,100*t2/tmp,M*N*8*disp_iter/t1/1024/1024,100*t1/tmp,M*J*N*8*disp_iter/t3/1024/1024,100*t3/tmp,100*t4/tmp,100*t5/tmp,100*t6/tmp,100*(t1+t2+t3+t4+t5+t6)/tmp,6500/it*t0,dkp)
        t1=0; t2=0; t3=0; t4=0; t5=0; t6=0;
        if dkp<1e-10; break; end
    end
end
time_Smol = toc;
%profile off
fprintf('N = %d\tmu = %d\ttime = %f\n',N,mu(1),time_Smol)
%profile report
if gpu>1; c=c{1}; end
c=gather(c);
%save(bfile,'b')

%% compute Euler equation errors
tic
T_test=10200; discard=200; Omega=chol(Sigma);
if gssa
    load Smolyak_Anisotropic_JMMV_2014/aT20200N10
    T=10000; x = [[ones(N,T_test); a20200(T+1:T+T_test,1:N)'] [1; nan(2*N-1,1)]];
else
    x=ones(2*N,T_test+1); rng(1); E=exp(Omega'*randn(N,T_test));
end
for t=1:T_test
    x(1:N,t+1)=c*smolyak(mu,xm,xs,S,x(:,t));
    if ~gssa
        x(N+1:2*N,t+1)=x(N+1:2*N,t).^rho.*E(:,t);
    end
end
x=x(:,1+discard:end);
T=T_test-discard;
k=x(1:N,1:T); kp=x(1:N,2:T+1); a=x(N+1:2*N,1:T);
ap=a.^rho.*e;
if gpu
    kernel.GridSize = ceil(T*J/kernel.ThreadBlockSize(1));
    x=[repmat(kp,1,1,J); ap];
    kpp=nan(N,T,J,'gpuArray');
    kpp = gather(feval(kernel, x, kpp, T*J, c));
else
    kpp=nan(N,T,J);
    for j=1:J
        kpp(:,:,j)=c*smolyak(mu,xm,xs,S,[kp; ap(:,:,j)]);
    end
end
% max(max(abs(kpp-kpp_)),[],3)
r=1-delta+A*alpha*kp.^(alpha-1).*ap;
ucp=sum((w.*mean(A*kp.^alpha.*ap-kpp+(1-delta)*kp).^-gam).*r,3);
uc=mean(A*a.*k.^alpha+(1-delta)*k-kp).^-gam;
err=1-beta*ucp./uc; % Unit-free Euler-equation errors
err_mean=log10(mean(abs(err(:))));
err_max=log10(max(abs(err(:))));
time_test = toc;
%% Display the results
format short g
disp(' '); disp('           SMOLYAK OUTPUT:'); disp(' '); 
disp('RUNNING TIME (in seconds):'); disp('');
disp('a) for computing the solution'); 
disp(time_Smol);
disp('b) for implementing the accuracy test'); 
disp(time_test);
disp('APPROXIMATION ERRORS (log10):'); disp(''); 
disp('a) mean Euler-equation error'); 
disp(err_mean)
disp('b) max Euler-equation error'); 
disp(err_max)
