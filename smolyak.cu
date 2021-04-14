// parameters needed for smolyak_kernel.cu: MU, N, D, M
// INPUTS:
// x - matrix of size D x L
// s - matrix of size MU x M in constant memory
// c - matrix of size N x M
// OUTPUT:
// y - matrix of size N x L

__constant__ double xm[D], xs[D];

#if MU*M<=65536
#define CONST_S
__constant__ unsigned char s[M][MU];
#endif

//#if (2*D+4*N)*M<=65536
//#define CONST_C
//__constant__ double c[M][N];
//#endif

#ifdef CONST_S
__global__ void smolyak(double const * __restrict__ x, double *y, int L, double const * __restrict__ c) {
#elif defined CONST_C
__global__ void smolyak(double const * __restrict__ x, double *y, int L) {
#else
__global__ void smolyak(double const * __restrict__ x, double *y, int L, double const * __restrict__ c, unsigned char const * __restrict__ s) {
#endif

	// Two options for thread-private storage p:
	// Option 1: keep p in local memory and set 48KB L1 cache for highest hit rate
	double p[1+D*(1<<MU)];
	// Option 2: shared memory, but kernel will be limited by 48KB shared memory capacity
//	__shared__ double p[D][1<<MU][blockDim.x]; // add [threadIdx.x] as 3rd index in p
	int i, j, t = threadIdx.x + blockIdx.x * blockDim.x;
	double b, yt[N] = {0.0};

//while (t<L) {
	if (t >= L) return;
	p[0] = 1.0;
	for (i = 0; i < D; i++) {
		double *pi = &p[1+i*(1<<MU)];
		pi[0] = b = (x[i+t*D]-xm[i])/xs[i];
		pi[1] = 2*b*b-1;
		for (j = 2; j < (1<<MU); j++)
			pi[j] = 2 * b * pi[j-1] - pi[j-2];
	}

	for (j = 0; j < M; j++) {
		b = 1.0;
		for (i = 0; i < MU; i++)
#ifdef CONST_S
			b *= p[s[j][i]];
#else
			b *= p[s[i+j*MU]];
#endif
		for (i = 0; i < N; i++)
#ifdef CONST_C
			yt[i] += b * c[j][i];
#else
			yt[i] += b * c[i+j*N];
#endif
	}
	for (i = 0; i < N; i++)
		y[i+t*N] = yt[i];
//	t += blockDim.x * gridDim.x;
//}
}

