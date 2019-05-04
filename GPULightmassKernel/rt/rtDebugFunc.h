namespace GPULightmass
{
	void LOG(const char* format, ...);
}
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <Windows.h>
#define cudaCheck(x) \
	{ \
		cudaError_t err = (x); \
		if (err != cudaSuccess) { \
			GPULightmass::LOG("Line %d: cudaCheckError: %s", __LINE__, cudaGetErrorString(err)); \
			MessageBoxA(0, cudaGetErrorString(err), "Error", 0); \
			assert(0); \
		} \
	}

#define cudaPostKernelLaunchCheck \
{ \
	cudaError_t err = cudaGetLastError(); \
	if (err != cudaSuccess) \
	{ \
		GPULightmass::LOG("PostKernelLaunchError: %s", cudaGetErrorString(err)); \
		MessageBoxA(0, cudaGetErrorString(err), "Error", 0); \
		assert(0); \
	} \
}