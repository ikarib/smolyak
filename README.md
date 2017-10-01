# smolyak
A Parallel Implementation of Smolyak Method

In this project, I show how to parallelize popular projection method called Smolyak algorithm involving sparse grids. The main hotspot in projection methods is the evaluation of a large polynomial on a large grid size. Fortunately, this problem turns out to be embarrassingly parallel. My program works in MATLAB by invoking a precompiled CUDA (Compute Unified Driver Architecture) kernel function as PTX (parallel thread execution) assembly for NVidia graphical processing units. This allows users to use their existing MATLAB codes without having to translate them into C language. I illustrate the practical application of my method by solving the same international real business cycle model with multiple countries. My algorithm improves performance in double precision by up to 66 times compared with serial implementation in Judd, Maliar, Maliar, and Valero's Smolyak toolbox also written in MATLAB. For example, ten country model with twenty states can be now solved with the third level of approximation in 1 hour and 3 minutes on Tesla K20c NVIDIA GPU rather than 70 hours on Intel CPU.
