#include <cuda_runtime.h>
#include "rt/rtDebugFunc.h"

#include "Radiosity.h"
#include "ProgressReport.h"

std::vector<SurfaceCacheDataPointers> SurfaceCaches;

void PreallocSurfaceCachePointers(const int Num)
{
	SurfaceCaches.resize(Num);
}

void CreateSurfaceCacheSampleData(
	const int ID,
	const int SizeX,
	const int SizeY,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float4 ReflectanceMap[],
	const float4 EmissiveMap[]
)
{
	static int NumSurfaceCacheImported = 0;

	SurfaceCaches[ID].SizeX = SizeX;
	SurfaceCaches[ID].SizeY = SizeY;
	SurfaceCaches[ID].GPUData.SizeX = SizeX;
	SurfaceCaches[ID].GPUData.SizeY = SizeY;

	cudaHostAlloc(&SurfaceCaches[ID].WorldPositionMap, SizeX * SizeY * sizeof(float4), 0);
	cudaHostAlloc(&SurfaceCaches[ID].WorldNormalMap, SizeX * SizeY * sizeof(float4), 0);
	cudaHostAlloc(&SurfaceCaches[ID].ReflectanceMap, SizeX * SizeY * sizeof(float4), 0);
	cudaHostAlloc(&SurfaceCaches[ID].EmissiveMap, SizeX * SizeY * sizeof(float4), 0);

	memcpy(SurfaceCaches[ID].WorldPositionMap, WorldPositionMap, SizeX * SizeY * sizeof(float4));
	memcpy(SurfaceCaches[ID].WorldNormalMap, WorldNormalMap, SizeX * SizeY * sizeof(float4));
	memcpy(SurfaceCaches[ID].ReflectanceMap, ReflectanceMap, SizeX * SizeY * sizeof(float4));
	memcpy(SurfaceCaches[ID].EmissiveMap, EmissiveMap, SizeX * SizeY * sizeof(float4));

	NumSurfaceCacheImported++;

	ReportProgress(
		"Importing surface cache " + std::to_string(NumSurfaceCacheImported) + "/" + std::to_string(SurfaceCaches.size()),
		(int)(NumSurfaceCacheImported * 100.0f / SurfaceCaches.size()),
		"",
		0
	);
}

void CreateSurfaceCacheTrasientDataOnGPU()
{
	for (int ID = 0; ID < SurfaceCaches.size(); ID++)
	{
		if (SurfaceCaches[ID].SizeX == 0)
			continue;

		cudaMalloc(&SurfaceCaches[ID].GPUData.cudaRadiositySurfaceCacheUnderlyingBuffer[0], SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4));
		cudaMalloc(&SurfaceCaches[ID].GPUData.cudaRadiositySurfaceCacheUnderlyingBuffer[1], SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4));
		cudaMalloc(&SurfaceCaches[ID].GPUData.cudaFinalLightingStagingBuffer, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4));
	}
}

std::vector<SurfaceCacheGPUDataPointers> GenerateSurfaceCacheGPUPointers()
{
	std::vector<SurfaceCacheGPUDataPointers> GPUPointers;
	GPUPointers.resize(SurfaceCaches.size());
	for (int ID = 0; ID < SurfaceCaches.size(); ID++)
	{
		if (SurfaceCaches[ID].SizeX == 0)
			continue;

		GPUPointers[ID].SizeX = SurfaceCaches[ID].GPUData.SizeX;
		GPUPointers[ID].SizeY = SurfaceCaches[ID].GPUData.SizeY;
		GPUPointers[ID].cudaRadiositySurfaceCacheUnderlyingBuffer[0] = SurfaceCaches[ID].GPUData.cudaRadiositySurfaceCacheUnderlyingBuffer[0];
		GPUPointers[ID].cudaRadiositySurfaceCacheUnderlyingBuffer[1] = SurfaceCaches[ID].GPUData.cudaRadiositySurfaceCacheUnderlyingBuffer[1];
		GPUPointers[ID].cudaFinalLightingStagingBuffer = SurfaceCaches[ID].GPUData.cudaFinalLightingStagingBuffer;
	}
	return GPUPointers;
}

void FreeSurfaceCacheTrasientData()
{
	for (int ID = 0; ID < SurfaceCaches.size(); ID++)
	{
		if (SurfaceCaches[ID].SizeX == 0)
			continue;

		cudaFree(SurfaceCaches[ID].GPUData.cudaRadiositySurfaceCacheUnderlyingBuffer[0]);
		cudaFree(SurfaceCaches[ID].GPUData.cudaRadiositySurfaceCacheUnderlyingBuffer[1]);
		cudaFreeHost(SurfaceCaches[ID].WorldPositionMap);
		cudaFreeHost(SurfaceCaches[ID].WorldNormalMap);
		cudaFreeHost(SurfaceCaches[ID].ReflectanceMap);
		cudaFreeHost(SurfaceCaches[ID].EmissiveMap);
	}
}

void UploadSurfaceCacheSampleDataToGPU(int ID)
{
	if (SurfaceCaches[ID].SizeX == 0)
		return;

	cudaCheck(cudaMalloc(&SurfaceCaches[ID].cudaWorldPositionMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4)));
	cudaCheck(cudaMalloc(&SurfaceCaches[ID].cudaWorldNormalMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4)));
	cudaCheck(cudaMalloc(&SurfaceCaches[ID].cudaReflectanceMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4)));
	cudaCheck(cudaMalloc(&SurfaceCaches[ID].cudaEmissiveMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4)));

	cudaMemcpyAsync(SurfaceCaches[ID].cudaWorldPositionMap, SurfaceCaches[ID].WorldPositionMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(SurfaceCaches[ID].cudaWorldNormalMap, SurfaceCaches[ID].WorldNormalMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(SurfaceCaches[ID].cudaReflectanceMap, SurfaceCaches[ID].ReflectanceMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(SurfaceCaches[ID].cudaEmissiveMap, SurfaceCaches[ID].EmissiveMap, SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4), cudaMemcpyHostToDevice);
}

void FreeSurfaceCacheSampleDataOnGPU(int ID)
{
	if (SurfaceCaches[ID].SizeX == 0)
		return;

	cudaFree(SurfaceCaches[ID].cudaWorldPositionMap);
	cudaFree(SurfaceCaches[ID].cudaWorldNormalMap);
	cudaFree(SurfaceCaches[ID].cudaReflectanceMap);
	cudaFree(SurfaceCaches[ID].cudaEmissiveMap);

	SurfaceCaches[ID].cudaWorldPositionMap = nullptr;
	SurfaceCaches[ID].cudaWorldNormalMap = nullptr;
	SurfaceCaches[ID].cudaReflectanceMap = nullptr;
	SurfaceCaches[ID].cudaEmissiveMap = nullptr;
}

size_t GetTotalVideoMemoryUsageDuringRadiosity()
{
	size_t TotalMemory = 0;
	for (int ID = 0; ID < SurfaceCaches.size(); ID++)
	{
		if (SurfaceCaches[ID].SizeX == 0)
			continue;

		TotalMemory += SurfaceCaches[ID].SizeX * SurfaceCaches[ID].SizeY * sizeof(float4) * 3;
	}
	return TotalMemory;
}
