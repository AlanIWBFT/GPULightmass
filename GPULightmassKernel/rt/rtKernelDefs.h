#pragma once

#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <cstdio>
#include <linear_math.h>
#include <helper_math.h>

#define USE_CORRELATED_SAMPLING 0

#define USE_JITTERED_SAMPLING 1
#define USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING 1

#define IRRADIANCE_CACHING_USE_ERROR_THRESHOLD 0
#define IRRADIANCE_CACHING_USE_DISTANCE 1
#define IRRADIANCE_CACHING_VISUALIZE 0
#define IRRADIANCE_CACHING_FORCE_NO_INTERPOLATION 0
#define IRRADIANCE_CACHING_DISTANCE_SCALE 1.0f
const int IRRADIANCE_CACHING_BASE_GRID_SPACING = 32;

struct SamplingGlobalParameters
{
	float FireflyClampingThreshold;
};

__device__ SamplingGlobalParameters GPUSamplingGlobalParameters;

// Old glory params ------------------------------------------------------------------

const int SampleCountOneDimension = 32;

const bool FORCE_SHADOWRAYS = false;

const int TaskBufferSize = 2048 * 2048;

__device__ cudaTextureObject_t BVHTreeNodesTexture; // float4
__device__ cudaTextureObject_t TriangleWoopCoordinatesTexture; // float4
__device__ cudaTextureObject_t MappingFromTriangleAddressToIndexTexture; // int

__device__ float4* BVHTreeNodes; // float4
__device__ float4* TriangleWoopCoordinates; // float4
__device__ int* MappingFromTriangleAddressToIndex; // int

__device__ cudaTextureObject_t SampleWorldPositionsTexture; // float4
__device__ cudaTextureObject_t SampleWorldNormalsTexture; // float4
__device__ cudaTextureObject_t TexelRadiusTexture; // float

__device__ cudaTextureObject_t SkyLightUpperHemisphereTexture; // float4
__device__ cudaTextureObject_t SkyLightLowerHemisphereTexture; // float4
__device__ cudaTextureObject_t SkyLightUpperHemisphereImportantDirectionsTexture; // int
__device__ cudaTextureObject_t SkyLightLowerHemisphereImportantDirectionsTexture; // int
__device__ cudaTextureObject_t SkyLightUpperHemisphereImportantColorTexture; // float4
__device__ cudaTextureObject_t SkyLightLowerHemisphereImportantColorTexture; // float4

__device__ int SkyLightCubemapNumThetaSteps;
__device__ int SkyLightCubemapNumPhiSteps;

__constant__ float2* VertexTextureUVs;
__constant__ float2* VertexTextureLightmapUVs;
__constant__ int* TriangleMappingIndex;
__constant__ int* TriangleIndexBuffer;

//__constant__ float4** GatheringRadiosityBuffers;
//__constant__ float4** ShootingRadiosityBuffers;
//__constant__ cudaTextureObject_t* ShootingRadiosityTextures;

__constant__ int BindedSizeX;
__constant__ int BindedSizeY;

namespace GPULightmass
{
struct GatheredLightSample
{
	SHVectorRGB SHVector;
	float SHCorrection;
	float3 IncidentLighting;
	float3 SkyOcclusion;
	float AverageDistance;
	float NumBackfaceHits;

	__device__ __host__ GatheredLightSample& PointLightWorldSpace(const float3 Color, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			SHVector.addIncomingRadiance(Color, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += (Color.x * 0.3f + Color.y * 0.59f + Color.z * 0.11f) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += Color * TangentDirection.z;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSample& PointLightWorldSpacePreweighted(const float3 PreweightedColor, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			float3 UnweightedRadiance = PreweightedColor / TangentDirection.z;
			SHVector.addIncomingRadiance(UnweightedRadiance, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += getLuminance(UnweightedRadiance) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += PreweightedColor;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSample operator*(float Scalar) const
	{
		GatheredLightSample Result;
		Result.SHVector = SHVector * Scalar;
		Result.SHCorrection = SHCorrection * Scalar;
		Result.IncidentLighting = IncidentLighting * Scalar;
		Result.SkyOcclusion = SkyOcclusion * Scalar;
		Result.AverageDistance = AverageDistance * Scalar;
		Result.NumBackfaceHits = NumBackfaceHits * Scalar;
		return Result;
	}

	__device__ __host__ GatheredLightSample& operator+=(const GatheredLightSample& rhs)
	{
		SHVector += rhs.SHVector;
		SHCorrection += rhs.SHCorrection;
		IncidentLighting += rhs.IncidentLighting;
		SkyOcclusion += rhs.SkyOcclusion;
		AverageDistance += rhs.AverageDistance;
		NumBackfaceHits += rhs.NumBackfaceHits;
		return *this;
	}

	__device__ __host__ GatheredLightSample operator+(const GatheredLightSample& rhs)
	{
		GatheredLightSample Result;
		Result.SHVector = SHVector + rhs.SHVector;
		Result.SHCorrection = SHCorrection + rhs.SHCorrection;
		Result.IncidentLighting = IncidentLighting + rhs.IncidentLighting;
		Result.SkyOcclusion = SkyOcclusion + rhs.SkyOcclusion;
		Result.AverageDistance = AverageDistance + rhs.AverageDistance;
		Result.NumBackfaceHits = NumBackfaceHits + rhs.NumBackfaceHits;
		return Result;
	}

	__device__ __host__ void Reset()
	{
		SHVector.r.reset();
		SHVector.g.reset();
		SHVector.b.reset();
		IncidentLighting = make_float3(0);
		SkyOcclusion = make_float3(0);
		SHCorrection = 0.0f;
		AverageDistance = 0.0f;
		NumBackfaceHits = 0;
	}
};

struct DirectionalLight
{
	float3 Color;
	float3 Direction;
};

struct PointLight
{
	float3 Color;
	float Radius;
	float3 WorldPosition;
};

struct SpotLight
{
	float3 Color;
	float Radius;
	float3 WorldPosition;
	float CosOuterConeAngle;
	float3 Direction;
	float CosInnerConeAngle;
};

}

__device__ GPULightmass::DirectionalLight* DirectionalLights;
__device__ GPULightmass::PointLight* PointLights;
__device__ GPULightmass::SpotLight* SpotLights;
__device__ int NumDirectionalLights;
__device__ int NumPointLights;
__device__ int NumSpotLights;

__device__ GPULightmass::GatheredLightSample* OutLightmapData;

int LaunchSizeX;
int LaunchSizeY;

__device__ int MappedTexelCounter;

__device__ int* IrradianceWorkspaceBuffer;

__align__(16)
struct TaskBuffer
{
	int Size;
	int Buffer[TaskBufferSize];
};


#include "../Radiosity.h"

__device__ int NumTotalSurfaceCaches;
__device__ const SurfaceCacheGPUDataPointers* RadiositySurfaceCaches;
__device__ const cudaTextureObject_t* MaskedCollisionMaps;