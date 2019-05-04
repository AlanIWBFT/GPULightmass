#pragma once
#include <vector>

struct SurfaceCacheGPUDataPointers
{
	int SizeX;
	int SizeY;
	float4* cudaRadiositySurfaceCacheUnderlyingBuffer[2] = { nullptr, nullptr };
	float4* cudaFinalLightingStagingBuffer = nullptr;
};

struct SurfaceCacheDataPointers
{
	int SizeX = 0;
	int SizeY = 0;
	float4* WorldPositionMap = nullptr;
	float4* WorldNormalMap = nullptr;
	float4* ReflectanceMap = nullptr;
	float4* EmissiveMap = nullptr;
	float4* cudaWorldPositionMap = nullptr;
	float4* cudaWorldNormalMap = nullptr;
	float4* cudaReflectanceMap = nullptr;
	float4* cudaEmissiveMap = nullptr;
	SurfaceCacheGPUDataPointers GPUData;
};

extern std::vector<SurfaceCacheDataPointers> SurfaceCaches;

void PreallocSurfaceCachePointers(const int Num);

void CreateSurfaceCacheSampleData(
	const int ID,
	const int SizeX,
	const int SizeY,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float4 ReflectanceMap[],
	const float4 EmissiveMap[]
);

void CreateSurfaceCacheTrasientDataOnGPU();

std::vector<SurfaceCacheGPUDataPointers> GenerateSurfaceCacheGPUPointers();

void FreeSurfaceCacheTrasientData();

void UploadSurfaceCacheSampleDataToGPU(int ID);

void FreeSurfaceCacheSampleDataOnGPU(int ID);

size_t GetTotalVideoMemoryUsageDuringRadiosity();
