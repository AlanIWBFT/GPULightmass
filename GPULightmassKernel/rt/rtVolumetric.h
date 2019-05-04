#pragma once
namespace GPULightmass
{

struct VolumetricLightSample
{
	SHVectorRGB3 SHVector;
	float3 IncidentLighting;
	float3 SkyOcclusion;
	float MinDistance;
	float BackfacingHitsFraction;

	__device__ __host__ VolumetricLightSample& PointLightWorldSpace(const float3 Color, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			SHVector.addIncomingRadiance(Color, 1, WorldDirection);
			IncidentLighting += Color * TangentDirection.z;
		}

		return *this;
	}

	__device__ __host__ VolumetricLightSample operator*(float Scalar) const
	{
		VolumetricLightSample Result;
		Result.SHVector = SHVector * Scalar;
		Result.IncidentLighting = IncidentLighting * Scalar;
		Result.SkyOcclusion = SkyOcclusion * Scalar;
		return Result;
	}

	__device__ __host__ VolumetricLightSample& operator+=(const VolumetricLightSample& rhs)
	{
		SHVector += rhs.SHVector;
		IncidentLighting += rhs.IncidentLighting;
		SkyOcclusion += rhs.SkyOcclusion;
		return *this;
	}

	__device__ __host__ VolumetricLightSample operator+(const VolumetricLightSample& rhs)
	{
		VolumetricLightSample Result;
		Result.SHVector = SHVector + rhs.SHVector;
		Result.IncidentLighting = IncidentLighting + rhs.IncidentLighting;
		Result.SkyOcclusion = SkyOcclusion + rhs.SkyOcclusion;
		return Result;
	}

	__device__ __host__ void Reset()
	{
		SHVector.r.reset();
		SHVector.g.reset();
		SHVector.b.reset();
		IncidentLighting = make_float3(0);
		SkyOcclusion = make_float3(0);
		MinDistance = 1e20f;
		BackfacingHitsFraction = 0.0f;
	}
};

}

const int NumVolumetricSamplesTheta = 32;
const int NumVolumetricSamplesPhi = 32 * 4;

const int VolumetricBlockSize = 16;

__global__ void VolumeSampleListIndirectLightingKernel(
	const int NumSamples,
	const float3 WorldPositions[],
	const float Direction,
	GPULightmass::VolumetricLightSample OutVolumeSamples[]
)
{
	int TargetTexelLocation = blockIdx.x;

	int threadId = TargetTexelLocation * threadIdx.y + threadIdx.x;
#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	curand_init(threadId, 0, 0, &randState);
#endif
#endif

	float Rand1 = 0.0f;
	float Rand2 = 0.0f;
	float Rand3 = 0.0f;

#if USE_JITTERED_SAMPLING
	Rand1 = 0.5f - curand_uniform(&randState);
	Rand2 = 0.5f - curand_uniform(&randState);
	Rand3 = 0.5f - curand_uniform(&randState);
#endif

	float3 WorldPosition = WorldPositions[TargetTexelLocation];
	float3 WorldNormal = make_float3(0.0f, 0.0f, Direction);

	float3 Tangent1, Tangent2;

	Tangent1 = cross(WorldNormal, make_float3(0, 0, 1));
	Tangent1 = length(Tangent1) < 0.1 ? cross(WorldNormal, make_float3(0, 1, 0)) : Tangent1;
	Tangent1 = normalize(Tangent1);
	Tangent2 = normalize(cross(Tangent1, WorldNormal));

	__shared__ GPULightmass::VolumetricLightSample GatheredRadiance[VolumetricBlockSize][VolumetricBlockSize];
	__shared__ int NumTotalHits[VolumetricBlockSize][VolumetricBlockSize];
	__shared__ int NumBackfaceHits[VolumetricBlockSize][VolumetricBlockSize];

	GatheredRadiance[threadIdx.x][threadIdx.y].Reset();

	NumBackfaceHits[threadIdx.x][threadIdx.y] = 0;
	NumTotalHits[threadIdx.x][threadIdx.y] = 0;

	for (int SampleIndexTheta = threadIdx.x; SampleIndexTheta < NumVolumetricSamplesTheta; SampleIndexTheta += VolumetricBlockSize)
	{
		for (int SampleIndexPhi = threadIdx.y; SampleIndexPhi <  NumVolumetricSamplesPhi; SampleIndexPhi += VolumetricBlockSize)
		{
			float RandA = 0.5f;
			float RandB = 0.5f;

#if USE_JITTERED_SAMPLING
			RandA = curand_uniform(&randState);
			RandB = curand_uniform(&randState);
#endif

			float U = 1.0f * (SampleIndexTheta + RandA) / NumVolumetricSamplesTheta;
			float Phi = 2.f * 3.1415926f * (SampleIndexPhi + RandB) / NumVolumetricSamplesPhi;
			float3 RayInLocalSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
			float3 RayInWorldSpace = normalize(Tangent1 * RayInLocalSpace.x + Tangent2 * RayInLocalSpace.y + WorldNormal * RayInLocalSpace.z);
			float3 RayOrigin = WorldPosition;

			HitInfo OutHitInfo;

			rtTrace(
				OutHitInfo,
				make_float4(RayOrigin, 0.01),
				make_float4(RayInWorldSpace, 1e20), false);

			if (OutHitInfo.TriangleIndex == -1)
			{
				GatheredRadiance[threadIdx.x][threadIdx.y].SkyOcclusion += RayInWorldSpace;
				GatheredRadiance[threadIdx.x][threadIdx.y].PointLightWorldSpace(fmaxf(SampleSkyLightRadiance(RayInWorldSpace), make_float3(0.0f)), RayInLocalSpace, RayInWorldSpace);
			}
			else
			{
				int I0 = TriangleIndexBuffer[OutHitInfo.TriangleIndex * 3 + 0];
				int I1 = TriangleIndexBuffer[OutHitInfo.TriangleIndex * 3 + 1];
				int I2 = TriangleIndexBuffer[OutHitInfo.TriangleIndex * 3 + 2];
				float2 UV0 = VertexTextureLightmapUVs[I0];
				float2 UV1 = VertexTextureLightmapUVs[I1];
				float2 UV2 = VertexTextureLightmapUVs[I2];
				float2 FinalUV = UV0 * OutHitInfo.TriangleUV.x + UV1 * OutHitInfo.TriangleUV.y + UV2 * (1.0f - OutHitInfo.TriangleUV.x - OutHitInfo.TriangleUV.y);

				int HitSurfaceCacheIndex = TriangleMappingIndex[OutHitInfo.TriangleIndex];

				int FinalY = FinalUV.y * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeY;
				int FinalX = FinalUV.x * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX;
				float4 RadiantExitance = RadiositySurfaceCaches[HitSurfaceCacheIndex].cudaFinalLightingStagingBuffer[FinalY * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX + FinalX];

				NumTotalHits[threadIdx.x][threadIdx.y]++;

				if (dot(OutHitInfo.TriangleNormalUnnormalized, RayInWorldSpace) < 0 || RadiantExitance.w > 0.5f)
				{
					GatheredRadiance[threadIdx.x][threadIdx.y].PointLightWorldSpace(make_float3(RadiantExitance), RayInLocalSpace, RayInWorldSpace);
				}
				else
				{
					NumBackfaceHits[threadIdx.x][threadIdx.y]++;
				}

				GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance = fminf(GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance, OutHitInfo.HitDistance);
			}
		}
	}

	GatheredRadiance[threadIdx.x][threadIdx.y] = GatheredRadiance[threadIdx.x][threadIdx.y] * (2.0f * 3.1415926f / (NumVolumetricSamplesTheta * NumVolumetricSamplesPhi));
	GatheredRadiance[threadIdx.x][threadIdx.y].SkyOcclusion /= 2.0f * 3.1415926f;

	for (int stride = VolumetricBlockSize >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();

		if (threadIdx.x < stride)
		{
			GatheredRadiance[threadIdx.x][threadIdx.y] += GatheredRadiance[threadIdx.x + stride][threadIdx.y];
			NumTotalHits[threadIdx.x][threadIdx.y] += NumTotalHits[threadIdx.x + stride][threadIdx.y];
			NumBackfaceHits[threadIdx.x][threadIdx.y] += NumBackfaceHits[threadIdx.x + stride][threadIdx.y];
			GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance = fminf(
				GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance,
				GatheredRadiance[threadIdx.x + stride][threadIdx.y].MinDistance);
		}

		__syncthreads();

		if (threadIdx.y < stride)
		{
			GatheredRadiance[threadIdx.x][threadIdx.y] += GatheredRadiance[threadIdx.x][threadIdx.y + stride];
			NumTotalHits[threadIdx.x][threadIdx.y] += NumTotalHits[threadIdx.x][threadIdx.y + stride];
			NumBackfaceHits[threadIdx.x][threadIdx.y] += NumBackfaceHits[threadIdx.x][threadIdx.y + stride];
			GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance = fminf(
				GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance,
				GatheredRadiance[threadIdx.x][threadIdx.y + stride].MinDistance);
		}
	}

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		GatheredRadiance[0][0].BackfacingHitsFraction = NumBackfaceHits[0][0] / NumTotalHits[0][0];
		OutVolumeSamples[TargetTexelLocation] = GatheredRadiance[0][0];
	}
}

__global__ void VolumetricBrickIndirectLightingKernel(
	const int BrickSize,
	const float3 WorldBrickMin,
	float3 WorldChildCellSize,
	const float Direction,
	GPULightmass::VolumetricLightSample OutVolumetricBrickData[]
)
{
	int TargetTexelLocation = blockIdx.x;
	int Z = blockIdx.x / (BrickSize * BrickSize);
	int Y = blockIdx.x % (BrickSize * BrickSize) / BrickSize;
	int X = blockIdx.x % (BrickSize * BrickSize) % BrickSize;

	int threadId = TargetTexelLocation * threadIdx.y + threadIdx.x;
#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	curand_init(threadId, 0, 0, &randState);
#endif
#endif

	float Rand1 = 0.0f;
	float Rand2 = 0.0f;
	float Rand3 = 0.0f;

#if USE_JITTERED_SAMPLING
	Rand1 = 0.5f - curand_uniform(&randState);
	Rand2 = 0.5f - curand_uniform(&randState);
	Rand3 = 0.5f - curand_uniform(&randState);
#endif

	float3 WorldPosition = WorldBrickMin + make_float3(X, Y, Z) * WorldChildCellSize;
	WorldPosition.x += 0.5f * WorldChildCellSize.x * Rand1;
	WorldPosition.y += 0.5f * WorldChildCellSize.y * Rand2;
	WorldPosition.z += 0.5f * WorldChildCellSize.z * Rand3;
	float3 WorldNormal = make_float3(0.0f, 0.0f, Direction);

	float3 Tangent1, Tangent2;

	Tangent1 = cross(WorldNormal, make_float3(0, 0, 1));
	Tangent1 = length(Tangent1) < 0.1 ? cross(WorldNormal, make_float3(0, 1, 0)) : Tangent1;
	Tangent1 = normalize(Tangent1);
	Tangent2 = normalize(cross(Tangent1, WorldNormal));

	__shared__ GPULightmass::VolumetricLightSample GatheredRadiance[VolumetricBlockSize][VolumetricBlockSize];
	__shared__ int NumTotalHits[VolumetricBlockSize][VolumetricBlockSize];
	__shared__ int NumBackfaceHits[VolumetricBlockSize][VolumetricBlockSize];

	GatheredRadiance[threadIdx.x][threadIdx.y].Reset();

	NumBackfaceHits[threadIdx.x][threadIdx.y] = 0;
	NumTotalHits[threadIdx.x][threadIdx.y] = 0;

	for (int SampleIndexTheta = threadIdx.x; SampleIndexTheta < NumVolumetricSamplesTheta; SampleIndexTheta += VolumetricBlockSize)
	{
		for (int SampleIndexPhi = threadIdx.y; SampleIndexPhi < NumVolumetricSamplesPhi; SampleIndexPhi += VolumetricBlockSize)
		{
			float RandA = 0.5f;
			float RandB = 0.5f;

		#if USE_JITTERED_SAMPLING
			RandA = curand_uniform(&randState);
			RandB = curand_uniform(&randState);
		#endif

			float U = 1.0f * (SampleIndexTheta + RandA) / NumVolumetricSamplesTheta;
			float Phi = 2.f * 3.1415926f * (SampleIndexPhi + RandB) / NumVolumetricSamplesPhi;
			float3 RayInLocalSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
			float3 RayInWorldSpace = normalize(Tangent1 * RayInLocalSpace.x + Tangent2 * RayInLocalSpace.y + WorldNormal * RayInLocalSpace.z);
			float3 RayOrigin = WorldPosition;

			HitInfo OutHitInfo;

			rtTrace(
				OutHitInfo,
				make_float4(RayOrigin, 0.01),
				make_float4(RayInWorldSpace, 1e20), false);

			if (OutHitInfo.TriangleIndex == -1)
			{
				GatheredRadiance[threadIdx.x][threadIdx.y].SkyOcclusion += RayInWorldSpace;
				GatheredRadiance[threadIdx.x][threadIdx.y].PointLightWorldSpace(fmaxf(SampleSkyLightRadiance(RayInWorldSpace), make_float3(0.0f)), RayInLocalSpace, RayInWorldSpace);
			}
			else
			{
				int I0 = TriangleIndexBuffer[OutHitInfo.TriangleIndex * 3 + 0];
				int I1 = TriangleIndexBuffer[OutHitInfo.TriangleIndex * 3 + 1];
				int I2 = TriangleIndexBuffer[OutHitInfo.TriangleIndex * 3 + 2];
				float2 UV0 = VertexTextureLightmapUVs[I0];
				float2 UV1 = VertexTextureLightmapUVs[I1];
				float2 UV2 = VertexTextureLightmapUVs[I2];
				float2 FinalUV = UV0 * OutHitInfo.TriangleUV.x + UV1 * OutHitInfo.TriangleUV.y + UV2 * (1.0f - OutHitInfo.TriangleUV.x - OutHitInfo.TriangleUV.y);

				int HitSurfaceCacheIndex = TriangleMappingIndex[OutHitInfo.TriangleIndex];

				int FinalY = FinalUV.y * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeY;
				int FinalX = FinalUV.x * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX;
				float4 RadiantExitance = RadiositySurfaceCaches[HitSurfaceCacheIndex].cudaFinalLightingStagingBuffer[FinalY * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX + FinalX];

				NumTotalHits[threadIdx.x][threadIdx.y]++;

				if (dot(OutHitInfo.TriangleNormalUnnormalized, RayInWorldSpace) < 0 || RadiantExitance.w > 0.5f)
				{
					GatheredRadiance[threadIdx.x][threadIdx.y].PointLightWorldSpace(make_float3(RadiantExitance), RayInLocalSpace, RayInWorldSpace);
				}
				else
				{
					NumBackfaceHits[threadIdx.x][threadIdx.y]++;
				}

				GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance = fminf(GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance, OutHitInfo.HitDistance);
			}
		}
	}

	GatheredRadiance[threadIdx.x][threadIdx.y] = GatheredRadiance[threadIdx.x][threadIdx.y] * (2.0f * 3.1415926f / (NumVolumetricSamplesTheta * NumVolumetricSamplesPhi));
	GatheredRadiance[threadIdx.x][threadIdx.y].SkyOcclusion /= 2.0f * 3.1415926f;

	for (int stride = VolumetricBlockSize >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();

		if (threadIdx.x < stride)
		{
			GatheredRadiance[threadIdx.x][threadIdx.y] += GatheredRadiance[threadIdx.x + stride][threadIdx.y];
			NumTotalHits[threadIdx.x][threadIdx.y] += NumTotalHits[threadIdx.x + stride][threadIdx.y];
			NumBackfaceHits[threadIdx.x][threadIdx.y] += NumBackfaceHits[threadIdx.x + stride][threadIdx.y];
			GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance = fminf(
				GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance,
				GatheredRadiance[threadIdx.x + stride][threadIdx.y].MinDistance);
		}

		__syncthreads();

		if (threadIdx.y < stride)
		{
			GatheredRadiance[threadIdx.x][threadIdx.y] += GatheredRadiance[threadIdx.x][threadIdx.y + stride];
			NumTotalHits[threadIdx.x][threadIdx.y] += NumTotalHits[threadIdx.x][threadIdx.y + stride];
			NumBackfaceHits[threadIdx.x][threadIdx.y] += NumBackfaceHits[threadIdx.x][threadIdx.y + stride];
			GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance = fminf(
				GatheredRadiance[threadIdx.x][threadIdx.y].MinDistance,
				GatheredRadiance[threadIdx.x][threadIdx.y + stride].MinDistance);
		}
	}

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		GatheredRadiance[0][0].BackfacingHitsFraction = NumBackfaceHits[0][0] / NumTotalHits[0][0];
		OutVolumetricBrickData[TargetTexelLocation] = GatheredRadiance[0][0];
	}
}

__host__ void rtLaunchVolumetric(
	const int BrickSize,
	const float3 WorldBrickMin,
	const float3 WorldChildCellSize,
	GPULightmass::VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	GPULightmass::VolumetricLightSample InOutVolumetricBrickLowerSamples[]
)
{
	float elapsedTime = 0.0f;

	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));

	cudaCheck(cudaEventRecord(startEvent, 0));

	{
		dim3 blockDim(VolumetricBlockSize, VolumetricBlockSize);
		dim3 gridDim(BrickSize * BrickSize * BrickSize, 1);
		VolumetricBrickIndirectLightingKernel << < gridDim, blockDim >> > (BrickSize, WorldBrickMin, WorldChildCellSize, 1.0f, InOutVolumetricBrickUpperSamples);
		cudaPostKernelLaunchCheck
	}

	{
		dim3 blockDim(VolumetricBlockSize, VolumetricBlockSize);
		dim3 gridDim(BrickSize * BrickSize * BrickSize, 1);
		VolumetricBrickIndirectLightingKernel << < gridDim, blockDim >> > (BrickSize, WorldBrickMin, WorldChildCellSize, -1.0f, InOutVolumetricBrickLowerSamples);
		cudaPostKernelLaunchCheck
	}

	cudaCheck(cudaEventRecord(stopEvent, 0));

	cudaCheck(cudaEventSynchronize(stopEvent));

	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	static double accumulatedGPUTime = 0.0f;
	accumulatedGPUTime += elapsedTime;

	//static int calculated = 0;

	//if(calculated % 64 == 0)
	//GPULightmass::LOG("GPU %dx%dx%d volumetric brick calculated in %.3fMS, total %.3fs", BrickSize, BrickSize, BrickSize, elapsedTime, accumulatedGPUTime / 1000.0);

	//calculated++;
}

__host__ void rtLaunchVolumeSamples(
	const int NumSamples,
	const float3 WorldPositions[],
	GPULightmass::VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	GPULightmass::VolumetricLightSample InOutVolumetricBrickLowerSamples[]
)
{
	float elapsedTime = 0.0f;

	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));

	cudaCheck(cudaEventRecord(startEvent, 0));

	{
		dim3 blockDim(VolumetricBlockSize, VolumetricBlockSize);
		dim3 gridDim(NumSamples, 1);
		VolumeSampleListIndirectLightingKernel <<< gridDim, blockDim >>> (NumSamples, WorldPositions, 1.0f, InOutVolumetricBrickUpperSamples);
		cudaPostKernelLaunchCheck
	}

	{
		dim3 blockDim(VolumetricBlockSize, VolumetricBlockSize);
		dim3 gridDim(NumSamples, 1);
		VolumeSampleListIndirectLightingKernel <<< gridDim, blockDim >>> (NumSamples, WorldPositions, -1.0f, InOutVolumetricBrickLowerSamples);
		cudaPostKernelLaunchCheck
	}

	cudaCheck(cudaEventRecord(stopEvent, 0));

	cudaCheck(cudaEventSynchronize(stopEvent));

	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	static double accumulatedGPUTime = 0.0f;
	accumulatedGPUTime += elapsedTime;

	//static int calculated = 0;

	//if(calculated % 64 == 0)
	//GPULightmass::LOG("GPU %dx%dx%d volumetric brick calculated in %.3fMS, total %.3fs", BrickSize, BrickSize, BrickSize, elapsedTime, accumulatedGPUTime / 1000.0);

	//calculated++;
}
