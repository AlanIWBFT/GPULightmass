#include "../sh_warpreduction.h"
#include "cub/cub.cuh"

struct SamplingParameters
{
	int NumSampleTheta;
	int NumSamplePhi;
	int TotalSamplePerTexel;

	int NumSamplePerBucketTheta;
	int NumSamplePerBucketPhi;
	int TotalSamplePerBucket;

	int ImageBlockSize;
};

__device__ SamplingParameters GPUSamplingParameters;

const int ImportanceImageFilteringFactor = 16;
const int MaxRayBufferSize = 64 * 64 * 4096;

const int NumBucketTheta = 16;
const int NumBucketPhi = NumBucketTheta;
const int TotalBucketPerTexel = NumBucketTheta * NumBucketPhi;

__device__ float4*	BucketRadiance;
__device__ int*		BucketSampleRejected;
__device__ float*	DownsampledBucketImportance;
__device__ int*		BucketRayStartOffsetInTexel;
__device__ int*		TexelToRayIDMap;

#define USE_ADAPTIVE_SAMPLING 1
#define VISUALIZE_ADAPTIVE_ONLY 0

#define SURFEL_EXPAND_FACTOR 1.5f

#include "rtTraceDynamicFetch.h"
#include "rtSurfelSorting.h"

__global__ void BucketRayGenKernel(
	const int Offset,
	const MortonHash6* InHashes
)
{
	const int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	if (Offset + threadId / GPUSamplingParameters.TotalSamplePerTexel >= MappedTexelCounter)
		return;
	int targetTexel = Offset + threadId / GPUSamplingParameters.TotalSamplePerTexel;
	
	targetTexel = InHashes[targetTexel].OriginalPosition;

	const int targetRay = threadId % GPUSamplingParameters.TotalSamplePerTexel;
	const int targetBucketInTexel = targetRay / GPUSamplingParameters.TotalSamplePerBucket;
	const int targetBucket = threadId / GPUSamplingParameters.TotalSamplePerBucket;
	const int targetRayInBucket = targetRay % GPUSamplingParameters.TotalSamplePerBucket;
	const int targetBucketTheta = targetBucketInTexel / NumBucketPhi;
	const int targetBucketPhi = targetBucketInTexel % NumBucketPhi;
	const int targetRayThetaInBucket = targetRayInBucket / GPUSamplingParameters.NumSamplePerBucketPhi;
	const int targetRayPhiInBucket = targetRayInBucket % GPUSamplingParameters.NumSamplePerBucketPhi;
	const int sampleIndexTheta = targetRayThetaInBucket + targetBucketTheta * GPUSamplingParameters.NumSamplePerBucketTheta;
	const int sampleIndexPhi = targetRayPhiInBucket + targetBucketPhi * GPUSamplingParameters.NumSamplePerBucketPhi;

	float3 worldPosition = make_float3(tex1Dfetch<float4>(SampleWorldPositionsTexture, targetTexel));
	float3 worldNormal = make_float3(tex1Dfetch<float4>(SampleWorldNormalsTexture, targetTexel));
	float texelRadius = tex1Dfetch<float>(TexelRadiusTexture, targetTexel);

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	curand_init(Offset * GPUSamplingParameters.TotalSamplePerTexel + threadId, 0, 0, &randState);
#endif
#endif

	float RandA = 0.5f;
	float RandB = 0.5f;
	float RandC = 0.5f;
	float RandD = 0.5f;

#if USE_JITTERED_SAMPLING
	RandA = curand_uniform(&randState);
	RandB = curand_uniform(&randState);
	RandC = curand_uniform(&randState);
	RandD = curand_uniform(&randState);
#endif

	float U = 1.0f * (sampleIndexTheta + RandA) / GPUSamplingParameters.NumSampleTheta;
	float Phi = 2.f * 3.1415926f * (sampleIndexPhi + RandB) / GPUSamplingParameters.NumSamplePhi;
	float3 RayInLocalSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
	//RayInLocalSpace = make_float3(sqrtf(fmaxf(0.0f, 1 - U)) * cos(Phi), sqrtf(fmaxf(0.0f, 1 - U)) * sin(Phi), sqrtf(U));
	float2 StratifiedPoint2D = make_float2((sampleIndexTheta + RandA) / GPUSamplingParameters.NumSampleTheta, (sampleIndexPhi + RandB) / GPUSamplingParameters.NumSamplePhi);
	float2 ConcentricMappedPoint = ConcentricSampleDisk(StratifiedPoint2D);
	RayInLocalSpace = LiftPoint2DToHemisphere(ConcentricMappedPoint);
#endif
	float3 RayInWorldSpace = normalize(tangent1 * RayInLocalSpace.x + tangent2 * RayInLocalSpace.y + worldNormal * RayInLocalSpace.z);

	float3 RayOrigin = worldPosition + worldNormal * texelRadius * 0.2f;
	RayOrigin += texelRadius * tangent1 * (0.5f - RandC) * SURFEL_EXPAND_FACTOR;
	RayOrigin += texelRadius * tangent2 * (0.5f - RandD) * SURFEL_EXPAND_FACTOR;

	Ray ray;
	ray.RayOriginAndNearClip = make_float4(RayOrigin, 0.01f);
	ray.RayDirectionAndFarClip = make_float4(RayInWorldSpace, 1e20);

	int rayID = threadId;
	atomicAggInc(&RayCount);

	if (targetRay == 0)
		TexelToRayIDMap[threadId / GPUSamplingParameters.TotalSamplePerTexel] = rayID;

	if (targetRayInBucket == 0)
		BucketRayStartOffsetInTexel[targetBucket] = GPUSamplingParameters.TotalSamplePerBucket * (1 + targetBucketInTexel);

	RayStartInfoBuffer[rayID].RayInWorldSpace = RayInWorldSpace;
	RayStartInfoBuffer[rayID].TangentZ = texelRadius == 0.0f ? -1.0f: RayInLocalSpace.z;
	RayHitResultBuffer[rayID].HitTriangleIndex = 0;

	RayBuffer[rayID] = ray;
}

__device__ float3 SampleRadiance(RayResult result, float3 RayInWorldSpace)
{
	if (result.HitTriangleIndex == -1)
	{
		return SampleSkyLightRadiance(RayInWorldSpace);
	}
	else
	{
		int I0 = TriangleIndexBuffer[result.HitTriangleIndex * 3 + 0];
		int I1 = TriangleIndexBuffer[result.HitTriangleIndex * 3 + 1];
		int I2 = TriangleIndexBuffer[result.HitTriangleIndex * 3 + 2];
		float2 UV0 = VertexTextureLightmapUVs[I0];
		float2 UV1 = VertexTextureLightmapUVs[I1];
		float2 UV2 = VertexTextureLightmapUVs[I2];
		float2 FinalUV = UV0 * result.TriangleUV.x + UV1 * result.TriangleUV.y + UV2 * (1.0f - result.TriangleUV.x - result.TriangleUV.y);

		int HitSurfaceCacheIndex = TriangleMappingIndex[result.HitTriangleIndex];

		int FinalY = FinalUV.y * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeY;
		int FinalX = FinalUV.x * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX;
		int LinearIndex = FinalY * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX + FinalX;
		LinearIndex = clamp(LinearIndex, 0, RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeY - 1);
		float4 RadiantExitance = RadiositySurfaceCaches[HitSurfaceCacheIndex].cudaFinalLightingStagingBuffer[LinearIndex];

		if (dot(result.TriangleNormalUnnormalized, RayInWorldSpace) < 0 || RadiantExitance.w > 0.5f)
		{
			return make_float3(RadiantExitance);
		}
		else
		{
			// Backface hit
			return make_float3(-1.0f);
		}
	}
}

__global__ void BucketGatherKernel()
{
	int targetBucket = 2 * blockIdx.x + threadIdx.y;
	if (targetBucket >= GPUSamplingParameters.ImageBlockSize * TotalBucketPerTexel) return;

	int targetTexel = targetBucket / TotalBucketPerTexel;
	int targetBucketInTexel = targetBucket % TotalBucketPerTexel;

	int totalRayInThisBucket;

	if (targetBucketInTexel == 0)
	{
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket];
	}
	else
	{
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket] - BucketRayStartOffsetInTexel[targetBucket - 1];
	}
	int rayStart = targetTexel * GPUSamplingParameters.TotalSamplePerTexel + (targetBucketInTexel == 0 ? 0 : BucketRayStartOffsetInTexel[targetBucket - 1]);
	int rayEnd = rayStart + totalRayInThisBucket;

	float4 bucketRadiance = make_float4(0.0f);

	int bucketSampleRejected = 0;

	for (int offset = 0; offset < totalRayInThisBucket; offset += warpSize)
	{
		int rayID = rayStart + offset + threadIdx.x;

		float3 radiance = make_float3(0);

		if (rayID < rayEnd)
		{
		#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
			radiance = RayStartInfoBuffer[rayID].TangentZ > 0.0f ? SampleRadiance(RayHitResultBuffer[rayID], RayStartInfoBuffer[rayID].RayInWorldSpace) : make_float3(0.0f);
			if (getLuminance(radiance) > GPUSamplingGlobalParameters.FireflyClampingThreshold)
				radiance = radiance / getLuminance(radiance) * GPUSamplingGlobalParameters.FireflyClampingThreshold;
		#else
			radiance = RayStartInfoBuffer[rayID].TangentZ > 0.0f ? SampleRadiance(RayHitResultBuffer[rayID], RayStartInfoBuffer[rayID].RayInWorldSpace) * RayStartInfoBuffer[rayID].TangentZ : make_float3(0.0f);
		#endif

			if (radiance.x < 0)
			{
				radiance = make_float3(0, 0, 0);
				bucketSampleRejected++;
			}
		}

		typedef cub::WarpReduce<float> WarpReduce;
		__shared__ typename WarpReduce::TempStorage temp_storage;

		radiance.x = WarpReduce(temp_storage).Sum(radiance.x);
		radiance.y = WarpReduce(temp_storage).Sum(radiance.y);
		radiance.z = WarpReduce(temp_storage).Sum(radiance.z);

		float variance = 0.0f;

		if (rayID < rayEnd)
		{
			float luminance = getLuminance(radiance);


			if (lane_id() % GPUSamplingParameters.NumSamplePerBucketPhi != GPUSamplingParameters.NumSamplePerBucketPhi - 1)
			{
				float luminanceNeighbour = __shfl_down_sync(__activemask(), luminance, 1);
				variance += fabsf(luminance - luminanceNeighbour);
			}

			if (lane_id() / GPUSamplingParameters.NumSamplePerBucketPhi != GPUSamplingParameters.NumSamplePerBucketTheta - 1)
			{
				float luminanceNeighbour = __shfl_down_sync(__activemask(), luminance, GPUSamplingParameters.NumSamplePerBucketPhi);
				variance += fabsf(luminance - luminanceNeighbour);
			}
		}

		variance = WarpReduce(temp_storage).Sum(variance);

		bucketRadiance += make_float4(radiance, variance);
	}

	{
		typedef cub::WarpReduce<int> WarpReduce;
		__shared__ typename WarpReduce::TempStorage temp_storage;
		bucketSampleRejected = WarpReduce(temp_storage).Sum(bucketSampleRejected);
	}

	if (threadIdx.x == 0)
	{
	#if VISUALIZE_ADAPTIVE_ONLY
		BucketRadiance[targetBucket] = bucketRadiance;
	#else
		BucketRadiance[targetBucket] += bucketRadiance;
	#endif

		BucketSampleRejected[targetBucket] += bucketSampleRejected;
	}
}

__device__ inline int mod(int a, int b)
{
	return (a%b+b)%b;
}
__global__ void DownsampleImportanceKernel()
{
	int targetTexel = blockIdx.x;
	if (targetTexel >= GPUSamplingParameters.ImageBlockSize / ImportanceImageFilteringFactor) return;
	int targetBucket = targetTexel * TotalBucketPerTexel + threadIdx.x;
	if (targetBucket >= GPUSamplingParameters.ImageBlockSize / ImportanceImageFilteringFactor * TotalBucketPerTexel) return;

	int targetBucketInTexel = threadIdx.x;
	const int targetBucketTheta = targetBucketInTexel / NumBucketPhi;
	const int targetBucketPhi = targetBucketInTexel % NumBucketPhi;

	float importance = 0.0001f;

	for (int i = 0; i < ImportanceImageFilteringFactor; i++)
	{
		if (ImportanceImageFilteringFactor * targetTexel + i < GPUSamplingParameters.ImageBlockSize)
		{
			int sourceBucket = (ImportanceImageFilteringFactor * targetTexel + i) * TotalBucketPerTexel + threadIdx.x;
			importance += BucketRadiance[sourceBucket].w;
			sourceBucket = (ImportanceImageFilteringFactor * targetTexel + i) * TotalBucketPerTexel + targetBucketTheta * NumBucketPhi + min(targetBucketPhi + 1, NumBucketPhi - 1);
			importance += BucketRadiance[sourceBucket].w;
			sourceBucket = (ImportanceImageFilteringFactor * targetTexel + i) * TotalBucketPerTexel + targetBucketTheta * NumBucketPhi + max(targetBucketPhi - 1, 0);
			importance += BucketRadiance[sourceBucket].w;
			sourceBucket = (ImportanceImageFilteringFactor * targetTexel + i) * TotalBucketPerTexel + max(targetBucketTheta - 1, 0) * NumBucketPhi + targetBucketPhi;
			importance += BucketRadiance[sourceBucket].w;
			sourceBucket = (ImportanceImageFilteringFactor * targetTexel + i) * TotalBucketPerTexel + min(targetBucketTheta + 1, NumBucketTheta - 1) * NumBucketPhi + targetBucketPhi;
			importance += BucketRadiance[sourceBucket].w;
		}
	}

	typedef cub::BlockReduce<float, TotalBucketPerTexel> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	
	float totalImportanceOnThread0 = BlockReduce(temp_storage).Sum(importance);
	
	__shared__ float totalImportance;

	if (threadIdx.x == 0)
		totalImportance = totalImportanceOnThread0;

	__syncthreads();

	DownsampledBucketImportance[targetBucket] = importance / totalImportance;

}

__global__ void ScatterImportanceAndCalculateSampleNumKernel()
{
	int targetBucket = blockIdx.x * TotalBucketPerTexel + threadIdx.x;
	if (targetBucket >= GPUSamplingParameters.ImageBlockSize * TotalBucketPerTexel) return;

	float importance = DownsampledBucketImportance[(blockIdx.x / ImportanceImageFilteringFactor) * TotalBucketPerTexel + threadIdx.x];

	int numSamplesThisBucket = importance * GPUSamplingParameters.TotalSamplePerTexel;

	int numSamplesBeforeThisBucket = 0;

	typedef cub::BlockScan<int, TotalBucketPerTexel> BlockScan;
	__shared__ typename BlockScan::TempStorage temp_storage;
	BlockScan(temp_storage).InclusiveSum(numSamplesThisBucket, numSamplesBeforeThisBucket);
	
	numSamplesBeforeThisBucket = min(numSamplesBeforeThisBucket, GPUSamplingParameters.TotalSamplePerTexel);

	BucketRayStartOffsetInTexel[targetBucket] = numSamplesBeforeThisBucket;
}

__global__ void BucketAdaptiveRayGenKernel(
	const int Offset,
	MortonHash6* InHashes
)
{
	int targetBucket = 2 * blockIdx.x + threadIdx.y;
	if (targetBucket >= GPUSamplingParameters.ImageBlockSize * TotalBucketPerTexel) return;
	int targetBucketInTexel = targetBucket % TotalBucketPerTexel;
	int targetTexel = targetBucket / TotalBucketPerTexel;

	if (Offset + targetTexel >= MappedTexelCounter)
		return;

	int targetTexelGlobal = InHashes[Offset + targetTexel].OriginalPosition;

	float3 worldPosition = make_float3(tex1Dfetch<float4>(SampleWorldPositionsTexture, targetTexelGlobal));
	float3 worldNormal = make_float3(tex1Dfetch<float4>(SampleWorldNormalsTexture, targetTexelGlobal));
	float texelRadius = tex1Dfetch<float>(TexelRadiusTexture, targetTexelGlobal);

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	int threadId = blockDim.x * blockDim.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
	curand_init(Offset * GPUSamplingParameters.TotalSamplePerTexel + threadId, 0, 0, &randState);
#endif
#endif
	RayCount = GPUSamplingParameters.ImageBlockSize * GPUSamplingParameters.TotalSamplePerTexel;
	int totalRayInThisBucket;
	if (targetBucketInTexel == 0)
	{
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket];
	}
	else
	{
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket] - BucketRayStartOffsetInTexel[targetBucket - 1];
	}
	//if (totalRayInThisBucket > GPUSamplingParameters.TotalSamplePerTexel) asm("brkpt;");
	int rayStart = targetTexel * GPUSamplingParameters.TotalSamplePerTexel + (targetBucketInTexel == 0 ? 0 : BucketRayStartOffsetInTexel[targetBucket - 1]);
	int rayEnd = rayStart + totalRayInThisBucket;

	int numSampleTheta = sqrtf(totalRayInThisBucket);
	int numSamplePhi = numSampleTheta;
	int numActualSamplesTaken = numSampleTheta * numSamplePhi;

	for (int offset = 0; offset < totalRayInThisBucket; offset += warpSize)
	{
		int RayIdInBucket = offset + threadIdx.x;
		int rayID = rayStart + RayIdInBucket;
		if (rayID < rayEnd)
		{
			const int targetBucketTheta = targetBucketInTexel / NumBucketPhi;
			const int targetBucketPhi = targetBucketInTexel % NumBucketPhi;

			float RandA = 0.5f;
			float RandB = 0.5f;
			float RandC = 0.5f;
			float RandD = 0.5f;

#if USE_JITTERED_SAMPLING
			RandA = 1.0f - curand_uniform(&randState);
			RandB = 1.0f - curand_uniform(&randState);
			RandC = 1.0f - curand_uniform(&randState);
			RandD = 1.0f - curand_uniform(&randState);
#endif
			
		#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
			//RayInLocalSpace = make_float3(sqrtf(fmaxf(0.0f, 1 - U)) * cos(Phi), sqrtf(fmaxf(0.0f, 1 - U)) * sin(Phi), sqrtf(U));
			float2 StratifiedPoint2D;
			if (RayIdInBucket < numActualSamplesTaken)
			{
				const int targetRayThetaInBucket = RayIdInBucket / numSamplePhi;
				const int targetRayPhiInBucket = RayIdInBucket % numSamplePhi;
				StratifiedPoint2D = make_float2(((targetRayThetaInBucket + RandA) / numSampleTheta + targetBucketTheta) / NumBucketTheta, ((targetRayPhiInBucket + RandB) / numSamplePhi + targetBucketPhi) / NumBucketPhi);
			}
			else
				StratifiedPoint2D = make_float2((RandA + targetBucketTheta) / NumBucketTheta, (RandB + targetBucketPhi) / NumBucketPhi);
			float2 ConcentricMappedPoint = ConcentricSampleDisk(StratifiedPoint2D);
			float3 RayInLocalSpace = LiftPoint2DToHemisphere(ConcentricMappedPoint);
		#endif
			float3 RayInWorldSpace = normalize(tangent1 * RayInLocalSpace.x + tangent2 * RayInLocalSpace.y + worldNormal * RayInLocalSpace.z);

			float3 RayOrigin = worldPosition + worldNormal * texelRadius * 0.2f;
			RayOrigin += texelRadius * tangent1 * (0.5f - RandC) * SURFEL_EXPAND_FACTOR;
			RayOrigin += texelRadius * tangent2 * (0.5f - RandD) * SURFEL_EXPAND_FACTOR;

			Ray ray;
			ray.RayOriginAndNearClip = make_float4(RayOrigin, 0.01f);
			ray.RayDirectionAndFarClip = make_float4(RayInWorldSpace, 1e20);

			if (targetBucketInTexel == 0 && RayIdInBucket == 0)
				TexelToRayIDMap[targetTexel] = rayID;

			RayBuffer[rayID] = ray;
			RayStartInfoBuffer[rayID].RayInWorldSpace = RayInWorldSpace;
			RayStartInfoBuffer[rayID].TangentZ = texelRadius == 0.0f ? -1.0f : RayInLocalSpace.z;
			RayHitResultBuffer[rayID].HitTriangleIndex = 0;
		}
	}
}

__global__ void BucketShadingKernel(
	const int Offset,
	MortonHash6* InHashes)
{
	int targetTexel = Offset + blockIdx.x;

	if (targetTexel >= MappedTexelCounter)
		return;

	targetTexel = InHashes[targetTexel].OriginalPosition;

	float3 worldPosition = make_float3(tex1Dfetch<float4>(SampleWorldPositionsTexture, targetTexel));
	bool isTwoSided = tex1Dfetch<float4>(SampleWorldPositionsTexture, targetTexel).w == 1.0f;
	float3 worldNormal = make_float3(tex1Dfetch<float4>(SampleWorldNormalsTexture, targetTexel));
	float texelRadius = tex1Dfetch<float>(TexelRadiusTexture, targetTexel);

	if (texelRadius == 0.0f)
		return;

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

	GPULightmass::GatheredLightSample LightSample;
	LightSample.Reset();

	const int sampleIndexTheta = threadIdx.x;
	const int sampleIndexPhi = threadIdx.y;

	float U = 1.0f * (sampleIndexTheta + 0.5f) / NumBucketTheta;
	float Phi = 2.f * 3.1415926f * (sampleIndexPhi + 0.5f) / NumBucketPhi;

	float3 RayInLocalSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
	//RayInLocalSpace = make_float3(sqrtf(fmaxf(0.0f, 1 - U)) * cos(Phi), sqrtf(fmaxf(0.0f, 1 - U)) * sin(Phi), sqrtf(U));
	float2 StratifiedPoint2D = make_float2((sampleIndexTheta + 0.5f) / NumBucketTheta, (sampleIndexPhi + 0.5f) / NumBucketPhi);
	float2 ConcentricMappedPoint = ConcentricSampleDisk(StratifiedPoint2D);
	RayInLocalSpace = LiftPoint2DToHemisphere(ConcentricMappedPoint);
#endif
	float3 RayInWorldSpace = normalize(tangent1 * RayInLocalSpace.x + tangent2 * RayInLocalSpace.y + worldNormal * RayInLocalSpace.z);

	int targetBucketInTexel = threadIdx.x * NumBucketPhi + threadIdx.y;
	int targetBucket = blockIdx.x * TotalBucketPerTexel + targetBucketInTexel;
	int totalRayInThisBucket;
	if (targetBucketInTexel == 0)
	{
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket];
	}
	else {
		totalRayInThisBucket = BucketRayStartOffsetInTexel[targetBucket] - BucketRayStartOffsetInTexel[targetBucket - 1];
	}

	int numActualSamplesTaken = 0;

#if USE_ADAPTIVE_SAMPLING
	numActualSamplesTaken = totalRayInThisBucket;
#endif

#if !VISUALIZE_ADAPTIVE_ONLY
	numActualSamplesTaken += GPUSamplingParameters.TotalSamplePerBucket;
#endif

	float3 radiance = (2 * 3.1415926f) * make_float3(BucketRadiance[(TotalBucketPerTexel * blockIdx.x + threadIdx.x * NumBucketPhi + threadIdx.y)]) / (max(numActualSamplesTaken - BucketSampleRejected[targetBucket], 1)) / TotalBucketPerTexel;
#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
	radiance *= 0.5f;
#endif

	LightSample.PointLightWorldSpacePreweighted(radiance, RayInLocalSpace, RayInWorldSpace);

	if (isTwoSided)
		LightSample.PointLightWorldSpacePreweighted(radiance, RayInLocalSpace, -RayInWorldSpace);

	LightSample.NumBackfaceHits = (float)BucketSampleRejected[targetBucket] / numActualSamplesTaken / TotalBucketPerTexel;

	LightSample = blockReduceSumToThread0(LightSample);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		OutLightmapData[targetTexel] = LightSample;
	}
}

__global__ void SkylightImportanceSampleKernel(
	const int Offset,
	MortonHash6* InHashes)
{
	int targetTexel = Offset + blockIdx.x;

	if (targetTexel >= MappedTexelCounter)
		return;

	targetTexel = InHashes[targetTexel].OriginalPosition;

	float3 worldPosition = make_float3(tex1Dfetch<float4>(SampleWorldPositionsTexture, targetTexel));
	bool isTwoSided = tex1Dfetch<float4>(SampleWorldPositionsTexture, targetTexel).w == 1.0f;
	float3 worldNormal = make_float3(tex1Dfetch<float4>(SampleWorldNormalsTexture, targetTexel));
	float texelRadius = tex1Dfetch<float>(TexelRadiusTexture, targetTexel);

	if (texelRadius == 0.0f)
		return;

	float3 tangent1, tangent2;

	tangent1 = cross(worldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(worldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, worldNormal));

	GPULightmass::GatheredLightSample LightSample;
	LightSample.Reset();

	const float SkylightCubemapResolution = 2.0f * SkyLightCubemapNumThetaSteps * SkyLightCubemapNumPhiSteps;
	const float PDF = 1 / (4 * 3.1415926f) * (SkylightCubemapResolution);

	int threadId = threadIdx.y * blockDim.x + threadIdx.x;
#if USE_JITTERED_SAMPLING
	curandState randState;
	curand_init(GPUSamplingParameters.ImageBlockSize + GPUSamplingParameters.TotalSamplePerTexel + Offset * GPUSamplingParameters.TotalSamplePerTexel + blockIdx.x * blockDim.x * blockDim.y + threadId, 0, 0, &randState);
#endif
	float3 skyLightNEERadiance = make_float3(0);

	if (threadId < 256)
	{
		const int sampleDirection = tex1Dfetch<int>(SkyLightUpperHemisphereImportantDirectionsTexture, threadId / 16);
		const int sampleTheta = sampleDirection / SkyLightCubemapNumPhiSteps;
		const int samplePhi = sampleDirection % SkyLightCubemapNumPhiSteps;
		const int subsampleTheta = (threadId % 16) / 4;
		const int subsamplePhi = (threadId % 16) % 4;
		float RandA = 0.5f;
		float RandB = 0.5f;
		float RandC = 0.5f;
		float RandD = 0.5f;
	#if USE_JITTERED_SAMPLING
		RandA = curand_uniform(&randState);
		RandB = curand_uniform(&randState);
		RandC = curand_uniform(&randState);
		RandD = curand_uniform(&randState);
	#endif
		float U = 1.0f * (sampleTheta + (subsampleTheta + 0.5f - RandA) / 4) / SkyLightCubemapNumThetaSteps;
		float Phi = 2.f * 3.1415926f * (samplePhi + (subsamplePhi + 0.5f - RandB) / 4) / SkyLightCubemapNumPhiSteps;
		float3 RayOrigin = worldPosition + worldNormal * texelRadius * 0.5f;
		RayOrigin += texelRadius * tangent1 * (0.5f - RandC) * 0.5f;
		RayOrigin += texelRadius * tangent2 * (0.5f - RandD) * 0.5f;
		float3 RayInWorldSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
		float3 RayInLocalSpace =  normalize(make_float3(dot(RayInWorldSpace, tangent1), dot(RayInWorldSpace, tangent2), dot(RayInWorldSpace, worldNormal)));
		if (RayInLocalSpace.z > 0)
		{
			HitInfo OutHitInfo;
			rtTrace(OutHitInfo, make_float4(RayOrigin, 0), make_float4(RayInWorldSpace, 1e20), true);
			if (OutHitInfo.TriangleIndex == -1)
			{
				skyLightNEERadiance = make_float3(tex1Dfetch<float4>(SkyLightUpperHemisphereImportantColorTexture, threadId / 16)) / PDF / 16;
				LightSample.PointLightWorldSpace(skyLightNEERadiance, RayInLocalSpace, RayInWorldSpace);
			}
		}
	}
	else if (threadId >= 256 && threadId < 512)
	{
		int sampleIndex = tex1Dfetch<int>(SkyLightLowerHemisphereImportantDirectionsTexture, threadId / 16 - 16);
		const int sampleTheta = sampleIndex / SkyLightCubemapNumPhiSteps;
		const int samplePhi = sampleIndex % SkyLightCubemapNumPhiSteps;
		const int subsampleTheta = (threadId % 16) / 4;
		const int subsamplePhi = (threadId % 16) % 4;
		float RandA = 0.5f;
		float RandB = 0.5f;
		float RandC = 0.5f;
		float RandD = 0.5f;
	#if USE_JITTERED_SAMPLING
		RandA = curand_uniform(&randState);
		RandB = curand_uniform(&randState);
		RandC = curand_uniform(&randState);
		RandD = curand_uniform(&randState);
	#endif
		float U = 1.0f * (sampleTheta + (subsampleTheta + 0.5f - RandA) / 4) / SkyLightCubemapNumThetaSteps;
		float Phi = 2.f * 3.1415926f * (samplePhi + (subsamplePhi + 0.5f - RandB) / 4) / SkyLightCubemapNumPhiSteps;
		float3 RayOrigin = worldPosition + worldNormal * texelRadius * 0.5f;
		RayOrigin += texelRadius * tangent1 * (0.5f - RandC) * 0.5f;
		RayOrigin += texelRadius * tangent2 * (0.5f - RandD) * 0.5f;
		float3 RayInWorldSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
		RayInWorldSpace.z = -RayInWorldSpace.z;
		float3 RayInLocalSpace = normalize(make_float3(dot(RayInWorldSpace, tangent1), dot(RayInWorldSpace, tangent2), dot(RayInWorldSpace, worldNormal)));
		if (RayInLocalSpace.z > 0)
		{
			HitInfo OutHitInfo;
			rtTrace(OutHitInfo, make_float4(RayOrigin, 0), make_float4(RayInWorldSpace, 1e20), true);
			if (OutHitInfo.TriangleIndex == -1)
			{
				skyLightNEERadiance = make_float3(tex1Dfetch<float4>(SkyLightLowerHemisphereImportantColorTexture, threadId / 16 - 16)) / PDF / 16;
				LightSample.PointLightWorldSpace(skyLightNEERadiance, RayInLocalSpace, RayInWorldSpace);
			}
		}
	}

	LightSample = blockReduceSumToThread0(LightSample);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		OutLightmapData[targetTexel] += LightSample;
	}
}