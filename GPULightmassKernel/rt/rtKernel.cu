// Device source code
#include "rtKernelDefs.h"
#include "rtHelperFunc.h"
#include "rtDebugFunc.h"
#include "rtTraceFunc.h"
#include "rtWarpAggregation.h"
#include <float.h>

__device__ float3 SampleSkyLightRadiance(float3 RayInWorldSpace)
{
	if (RayInWorldSpace.z > 0)
	{
		// Upper hemisphere
		int ThetaIndex = RayInWorldSpace.z * SkyLightCubemapNumThetaSteps;
		float R = sqrtf(1.0f - RayInWorldSpace.z * RayInWorldSpace.z);
		float Phi = acosf(RayInWorldSpace.x / R);
		if (RayInWorldSpace.y < 0) Phi = 2.0f * 3.1415926f - Phi;
		int PhiIndex = Phi / (2.0f * 3.1415926f) * SkyLightCubemapNumPhiSteps;
		return make_float3(tex1Dfetch(SkyLightUpperHemisphereTexture, ThetaIndex * SkyLightCubemapNumPhiSteps + PhiIndex));
	}
	else
	{
		// Lower hemisphere
		int ThetaIndex = (-RayInWorldSpace.z) * SkyLightCubemapNumThetaSteps;
		float R = sqrtf(1.0f - RayInWorldSpace.z * RayInWorldSpace.z);
		float Phi = acosf(RayInWorldSpace.x / R);
		if (RayInWorldSpace.y < 0) Phi = 2.0f * 3.1415926f - Phi;
		int PhiIndex = Phi / (2.0f * 3.1415926f) * SkyLightCubemapNumPhiSteps;
		return make_float3(tex1Dfetch(SkyLightLowerHemisphereTexture, ThetaIndex * SkyLightCubemapNumPhiSteps + PhiIndex));
	}
}

__global__ void GenerateSignedDistanceFieldVolumeDataKernel(Vec3f BoundingBoxMin, Vec3f BoundingBoxMax, Vec3i VolumeDimension, float* OutBuffer, int ZSliceIndex)
{
	const int SampleCountOneDimension = 64;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = (blockIdx.z + ZSliceIndex) * blockDim.z + threadIdx.z;
	if (x >= VolumeDimension.x || y >= VolumeDimension.y || z >= VolumeDimension.z) return;
	const int threadId = (blockIdx.z * blockDim.z + threadIdx.z) * VolumeDimension.y * VolumeDimension.x + y * VolumeDimension.x + x;
	curandState randState;
	curand_init(threadId, 0, 0, &randState);
	const Vec3f VolumeSize = BoundingBoxMax - BoundingBoxMin;
	const Vec3f DistanceFieldVoxelSize = VolumeSize / Vec3f(VolumeDimension.x, VolumeDimension.y, VolumeDimension.z);
	const Vec3f VoxelPosition = Vec3f(x + .5f, y + .5f, z + .5f) * DistanceFieldVoxelSize + BoundingBoxMin;

	float MinDistance = VolumeSize.length();
	int Hit = 0;
	int HitBack = 0;

	for (int SampleIndexTheta = 0; SampleIndexTheta < SampleCountOneDimension; SampleIndexTheta++) {
		for (int SampleIndexPhi = 0; SampleIndexPhi < SampleCountOneDimension; SampleIndexPhi++) {
			const float U = -1.0f + 2.0f * (SampleIndexTheta + curand_uniform(&randState)) / SampleCountOneDimension;
			const float Phi = 2.f * 3.1415926f * (SampleIndexPhi + curand_uniform(&randState)) / SampleCountOneDimension;
			const float3 SampleDirection = make_float3(sqrt(1 - U * U) * cos(Phi), sqrt(1 - U * U) * sin(Phi), U);

			const float ray_tmin = 0.00001f; // set to 0.01f when using refractive material
			const float ray_tmax = VolumeSize.length();

			HitInfo OutHitInfo;

			rtTrace(OutHitInfo,
				make_float4(VoxelPosition.x, VoxelPosition.y, VoxelPosition.z, ray_tmin),
				make_float4(SampleDirection.x, SampleDirection.y, SampleDirection.z, ray_tmax), true);

			MinDistance = MinDistance < OutHitInfo.HitDistance ? MinDistance : OutHitInfo.HitDistance;

			if (OutHitInfo.TriangleIndex != -1)
			{
				Hit++;
				if (dot(OutHitInfo.TriangleNormalUnnormalized, SampleDirection) > 0)
					HitBack++;
			}
		}
	}

	if (HitBack >= SampleCountOneDimension * SampleCountOneDimension * .5f || (MinDistance < DistanceFieldVoxelSize.length() && HitBack > .95f * Hit))
		MinDistance = -MinDistance;

	OutBuffer[threadId] = MinDistance / VolumeSize.max();
}

struct Ray
{
	float4 RayOriginAndNearClip;
	float4 RayDirectionAndFarClip;
};

struct RayStartInfo
{
	float3 RayInWorldSpace;
	float TangentZ;
};

struct RayResult
{
	float3 TriangleNormalUnnormalized;
	int HitTriangleIndex;
	float2 TriangleUV;
};

const int BLOCK_DIM_X = 8;
const int BLOCK_DIM_Y = 8;

__device__ Ray* RayBuffer;// [ImageBlockSize * TotalSamplePerTexel];
__device__ RayStartInfo* RayStartInfoBuffer;// [ImageBlockSize * TotalSamplePerTexel];
__device__ RayResult* RayHitResultBuffer;// [ImageBlockSize * TotalSamplePerTexel];
__device__ RayResult* RayHitResultBufferUninterleaved;// [ImageBlockSize * TotalSamplePerTexel];
__device__ int RayCount = 0;
__device__ int FinishedRayCount = 0;

#include "rtRadiosity.h"
#include "rtVolumetric.h"
#include "rtRayBufferKernel.h"

// Host source code
#include "rtHostConfig.h"
