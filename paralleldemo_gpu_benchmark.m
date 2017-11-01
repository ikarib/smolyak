%% Measuring GPU Performance
% This example shows how to measure some of the key performance
% characteristics of a GPU.
%
% GPUs can be used to speed up certain types of computations.  However,
% GPU performance varies widely between different GPU devices.  In order to
% quantify the performance of a GPU, three tests are used:
%
% * How quickly can data be sent to the GPU or read back from it?
% * How fast can the GPU kernel read and write data?
% * How fast can the GPU perform computations?
%
% After measuring these, the performance of the GPU can be compared to the
% host CPU.  This provides a guide as to how much data or computation is 
% required for the GPU to provide an advantage over the CPU.

% Copyright 2013-2016 The MathWorks, Inc.

%% Setup
delete(gcp('nocreate'));
nGPUs = gpuDeviceCount;
if nGPUs>1
    myCluster = parcluster('local');
    myCluster.NumWorkers = nGPUs;
    saveProfile(myCluster);
    parpool('local', nGPUs);
end
spmd
    gpu = gpuDevice;
    fprintf('GPU%d: %s\n', gpu.Index, gpu.Name)
end

%% Testing host/GPU bandwidth
% The first test estimates how quickly data can be sent to and
% read from the GPU.  Because the GPU is plugged into the PCI bus, this
% largely depends on how fast the PCI bus is and how many other things are
% using it.  However, there are also some overheads that are included in
% the measurements, particularly the function call overhead and the array
% allocation time.  Since these are present in any "real world" use of the
% GPU, it is reasonable to include these.
%
% In the following tests, memory is allocated and data is sent to the GPU
% using the <matlab:doc('gpuArray') |gpuArray|> function.  Memory is
% allocated and data is transferred back to host memory using 
% <matlab:doc('gpuArray/gather') |gather|>.
%
% Note that PCI express v3, as used in this test, has a theoretical
% bandwidth of 0.99GB/s per lane. For the 16-lane slots (PCIe3 x16) used by
% NVIDIA's compute cards this gives a theoretical 15.75GB/s.

%% Testing computationally intensive operations
% For operations where the number of floating-point computations performed
% per element read from or written to memory is high, the memory speed is
% much less important.  In this case the number and speed of the
% floating-point units is the limiting factor.  These operations are said
% to have high "computational density".
%
% A good test of computational performance is a matrix-matrix multiply.
% For multiplying two $N \times N$ matrices, the total number of
% floating-point calculations is
%
% $FLOPS(N) = 2N^3 - N^2$.
%
% Two input matrices are read and one resulting matrix is
% written, for a total of $3N^2$ elements read or written.  This gives a
% computational density of |(2N - 1)/3| FLOP/element.  Contrast this with
% |plus| as used above, which has a computational density of |1/2|
% FLOP/element.
sizes = power(2, 20:2:28);
N = sqrt(sizes);
sendBandwidth = inf(nGPUs,numel(sizes));
mmGFlopsGPU = inf(nGPUs,numel(sizes));
gatherBandwidth = inf(nGPUs,numel(sizes));
for ii=1:numel(sizes)
    fprintf('Matrix size: %g\n',N(ii))
    A = rand( N(ii), N(ii) );
    spmd
        tmp=tic;
        a = gpuArray(A);
        sendTimes = toc(tmp);
%         mmTimesGPU(ii) = gputimeit(@() A*B);
        tmp=tic;
        B = a*a;
        wait(gpu)
        mmTimesGPU = toc(tmp);
        tmp=tic;
        gather(B);
        gatherTimes = toc(tmp);
    end
    mmGFlopsGPU(:,ii) = (2*N(ii).^3 - N(ii).^2)./vertcat(mmTimesGPU{:})/1e9;
    sendBandwidth(:,ii) = (sizes(ii)*8./vertcat(sendTimes{:}))/1e6;
    gatherBandwidth(:,ii) = (sizes(ii)*8./vertcat(gatherTimes{:}))/1e6;
    fprintf('GPU%d: %1.1f GFLOPS (send %g MB/s, gather %f MB/s)\n',[(1:nGPUs)' mmGFlopsGPU(:,ii) sendBandwidth(:,ii) gatherBandwidth(:,ii)]')
end
[maxGFlopsGPU,maxGFlopsGPUIdx] = max(mmGFlopsGPU,[],2);
[maxSendBandwidth,maxSendIdx] = max(sendBandwidth,[],2);
[maxGatherBandwidth,maxGatherIdx] = max(gatherBandwidth,[],2);

%%
% Now plot it to see where the peak was achieved.
hold off
semilogx(sizes, mmGFlopsGPU)
hold on
semilogx(sizes(maxGFlopsGPUIdx), maxGFlopsGPU, 'bo-', 'MarkerSize', 10);
grid on
title('Double precision matrix-matrix multiply')
xlabel('Matrix size (numel)')
ylabel('Calculation Rate (GFLOPS)')
legend(sprintfc('GPU%d',1:nGPUs), 'Location', 'NorthWest')


%% Conclusions
% These tests reveal some important characteristics of GPU performance:
%
% * Transfers from host memory to GPU memory and back are relatively slow.
% * A good GPU can read/write its memory much faster than the host CPU can
% read/write its memory.
% * Given large enough data, GPUs can perform calculations much faster than
% the host CPU.
%
% It is notable that in each test quite large arrays were required to fully
% saturate the GPU, whether limited by memory or by computation.  GPUs
% provide the greatest advantage when working with millions of elements at
% once.
%
% More detailed GPU benchmarks, including comparisons between different
% GPUs, are available in 
% <http://www.mathworks.com/matlabcentral/fileexchange/34080 GPUBench> on
% the <http://www.mathworks.com/matlabcentral/fileexchange MATLAB(R)
% Central File Exchange>.
