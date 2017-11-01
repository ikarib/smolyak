// parameters needed for smolyak_kernel.cu: D, L, M, N, SMAX, MU_MAX
// INPUTS:
// x - matrix of size L x D
// s - matrix of size M x MU_MAX in constant memory
// c - matrix of size M x N
// OUTPUT:
// y - matrix of size L x N

__constant__ double xm[D], xs[D];

#if MU_MAX*M<=65536
#define CONST_S
__constant__ unsigned char s[MU_MAX][M];
#endif

//#if (2*D+4*N)*M<=65536
//#define CONST_C
//__constant__ double c[N][M];
//#endif

#ifdef CONST_S
__global__ void smolyak(double const * __restrict__ x, double *y, double const * __restrict__ c) {
#elif defined CONST_C
__global__ void smolyak(double const * __restrict__ x, double *y) {
#else
__global__ void smolyak(double const * __restrict__ x, double *y, double const * __restrict__ c, unsigned char const * __restrict__ s) {
#endif

	// Two options for thread-private storage p:
	// Option 1: keep p in local memory and set 48KB L1 cache for highest hit rate
	double p[1+SMAX*D];
	// Option 2: shared memory, but kernel will be limited by 48KB shared memory capacity
//	__shared__ double p[D][SMAX][blockDim.x]; // add [threadIdx.x] as 3rd index in p
	int i, j, t = threadIdx.x + blockIdx.x * blockDim.x;
	double b, yt[N] = {0.0};

//	while (t<L) {
	if (t < L) {
		p[0] = 1.0;
		for (i = 0; i < D; i++) {
			double *pi = &p[1+i*SMAX];
			pi[0] = b = (x[t+i*L]-xm[i])/xs[i];
			pi[1] = 2*b*b-1;
			for (j = 2; j < SMAX; j++)
				pi[j] = 2 * b * pi[j-1] - pi[j-2];
		}

		for (j = 0; j < M; j++) {
			b = 1.0;
			for (i = 0; i < MU_MAX; i++)
#ifdef CONST_S
				b *= p[s[i][j]];
#else
				b *= p[s[j+i*M]];
#endif
			for (i = 0; i < N; i++)
#ifdef CONST_C
				yt[i] += b * c[i][j];
#else
				yt[i] += b * c[j+i*M];
#endif
		}
		for (i = 0; i < N; i++)
			y[t+i*L] = yt[i];
//		t += blockDim.x * gridDim.x;
	}
}

