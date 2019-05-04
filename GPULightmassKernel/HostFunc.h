#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "linear_math.h"

void BindBVHData(
	std::vector<float4>& NodeData,
	std::vector<float4>& WoopifiedTriangles,
	std::vector<int>& TriangleIndices);
void BindParameterizationData(
	const int NumVertices,
	const int NumTriangles,
	const float2 VertexTextureUVBuffer[],
	const float2 VertexLightmapUVBuffer[],
	const int3 TriangleIndexBuffer[],
	const int TriangleTextureMappingIndex[]);
void LaunchRadiosityLoop(int NumBounces, int NumSamplesFirstPass);
void BindSkyLightCubemap(
	const int NumThetaSteps,
	const int NumPhiSteps,
	const float4 InUpperHemisphereCubemap[],
	const float4 InLowerHemisphereCubemap[],
	const int UpperHemisphereImportantDirections[],
	const int LowerHemisphereImportantDirections[],
	const float4 InUpperHemisphereImportantColor[],
	const float4 InLowerHemisphereImportantColor[]
);
bool BindBVHDataFromFile(std::string fileName);
void SaveSampleDataToFile(const float4* SampleWorldPositions, const float4 * SampleWorldNormals, const float* TexelRadius, int SizeX, int SizeY, std::string fileName);
void CalculateLighting(
	const float4 * SamplingWorldPositions, 
	const float4 * SamplingWorldNormals,
	const float* TexelRadius, 
	GPULightmass::GatheredLightSample OutLightmapData[],
	int SizeX, int SizeY, 
	int NumSamples);
void GenerateSDF(Vec3f BoundingBoxMin, Vec3f BoundingBoxMax, Vec3i VolumeDimension, float* OutBuffer);

