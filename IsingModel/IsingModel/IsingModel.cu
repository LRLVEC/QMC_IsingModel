#include <_Time.h>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <curand_kernel.h>

template<class T, unsigned long long blockSize>__device__ void warpReduce(volatile T* sdata, unsigned int tid)
{
	if constexpr (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if constexpr (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if constexpr (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if constexpr (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if constexpr (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if constexpr (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}
template<class T, unsigned long long blockSize>__global__ void reduction(T* a, T* b, unsigned int N)//N % (2*blockSize) == 0
{
	unsigned int const tid(threadIdx.x);
	unsigned int i(blockIdx.x * blockSize * 2 + tid);
	unsigned int const gridSize(2 * blockSize * gridDim.x);
	T ans(0);
	while (i < N)
	{
		//if (i + blockSize < N)
		ans += a[i] + a[i + blockSize];
		//else
		//	ans += a[i];
		i += gridSize;
	}
	__shared__ T sdata[blockSize];//must fix it size and don't use extern!!!
	sdata[tid] = ans;
	__syncthreads();
	if constexpr (blockSize == 1024)
	{
		if (tid < 512)
			sdata[tid] += sdata[tid + 512];
		__syncthreads();
	}
	if constexpr (blockSize >= 512)
	{
		if (tid < 256)
			sdata[tid] += sdata[tid + 256];
		__syncthreads();
	}
	if constexpr (blockSize >= 256)
	{
		if (tid < 128)
			sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if constexpr (blockSize >= 128)
	{
		if (tid < 64)
			sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	/*if (tid < 32)
	{
		if constexpr (blockSize >= 64)ans = sdata[tid] + sdata[tid + 32];
		else ans = sdata[tid];
		if constexpr (blockSize >= 32)ans += __shfl_down_sync(0xffff, ans, 16);
		if constexpr (blockSize >= 16)ans += __shfl_down_sync(0xff, ans, 8);
		if constexpr (blockSize >= 8)ans += __shfl_down_sync(0xf, ans, 4);
		if constexpr (blockSize >= 4)ans += __shfl_down_sync(0x3, ans, 2);
		if constexpr (blockSize >= 2)ans += __shfl_down_sync(0x1, ans, 1);
	}*/
	if (tid < 32)warpReduce<T, blockSize>(sdata, tid);
	if (tid == 0)b[blockIdx.x] = sdata[0];
}
template<class T, unsigned long long blockSize>__global__ void reduction1(T* a, T* b, unsigned int N)//N % (4 * blockSize) == 0
{
	unsigned int const tid(threadIdx.x);
	unsigned int i(blockIdx.x * blockSize * 4 + tid);
	unsigned int const gridSize(4 * blockSize * gridDim.x);
	constexpr unsigned int warpNum(blockSize / 32);
	__shared__ T sdata[warpNum];//must fix it size and don't use extern!!!
	T ans(0);
	while (i < N)
	{
		ans += a[i + 3 * blockSize] + a[i + 2 * blockSize] + a[i + blockSize] + a[i];
		i += gridSize;
	}
	ans += __shfl_down_sync(0xffff, ans, 16);
	ans += __shfl_down_sync(0xff, ans, 8);
	ans += __shfl_down_sync(0xf, ans, 4);
	ans += __shfl_down_sync(0x3, ans, 2);
	ans += __shfl_down_sync(0x1, ans, 1);
	if ((tid & 31) == 0)sdata[tid / 32] = ans;
	__syncthreads();
	if (tid < 32)
	{
		if constexpr (warpNum >= 32)ans += __shfl_down_sync(0xffff, ans, 16);
		if constexpr (warpNum >= 16)ans += __shfl_down_sync(0xff, ans, 8);
		if constexpr (warpNum >= 8)ans += __shfl_down_sync(0xf, ans, 4);
		if constexpr (warpNum >= 4)ans += __shfl_down_sync(0x3, ans, 2);
		if constexpr (warpNum >= 2)ans += __shfl_down_sync(0x1, ans, 1);
		if (tid == 0)b[blockIdx.x] = ans;
	}
}


int main()
{
	Timer timer;
	std::mt19937 mt(time(0));
	std::uniform_int_distribution<unsigned int> rd(0, 5);

	constexpr unsigned long long N(1024llu * 1024llu * 2048llu);
	constexpr unsigned int gridDim(1024);
	constexpr unsigned long long aSize(N * sizeof(unsigned int));
	constexpr unsigned int bSize(gridDim * sizeof(unsigned int));

	unsigned int* a((unsigned int*)::malloc(aSize));
	if (a == nullptr)
	{
		::printf("malloc failed!");
		return -1;
	}
	unsigned int* aDevice;
	unsigned int* bDevice;
	unsigned int* answerDevice;
	unsigned int stdAnswer(0), answer;
	cudaMalloc(&aDevice, aSize);
	cudaMalloc(&bDevice, bSize);
	cudaMalloc(&answerDevice, sizeof(unsigned int));
	::printf("Init numbers:\n");
	for (unsigned long long c0(0); c0 < N; ++c0)a[c0] = 1;// rd(mt);
	timer.begin();
	for (unsigned long long c0(0); c0 < N; ++c0)stdAnswer += a[c0];
	timer.end();
	timer.print("CPU:");
	::printf("%u\n", stdAnswer);

	cudaMemcpy(aDevice, a, aSize, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	timer.begin();
	reduction1<unsigned int, 1024> << <1024, 1024 >> > (aDevice, bDevice, N - 1);
	reduction<unsigned int, 512> << <1, 512 >> > (bDevice, answerDevice, gridDim);
	cudaDeviceSynchronize();
	timer.end();
	timer.print("GPU:");
	cudaMemcpy(&answer, answerDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	::printf("%u\n", answer);

	cudaFree(aDevice);
	cudaFree(bDevice);
	cudaFree(answerDevice);
	free(a);
}