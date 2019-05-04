#pragma once

__device__ int NumRadiositySampleFirstPass;

__global__ void RadiosityBaseGridGenerationKernel(
	const float4 * WorldNormalMap,
	const float4 * EmissiveMap,
	const int SizeX,
	const int SizeY,
	TaskBuffer* OutTaskBuffer)
{
	int X = (blockIdx.x * blockDim.x + threadIdx.x) * IRRADIANCE_CACHING_BASE_GRID_SPACING;
	int Y = (blockIdx.y * blockDim.y + threadIdx.y) * IRRADIANCE_CACHING_BASE_GRID_SPACING;
	if (X >= SizeX || Y >= SizeY)
		return;
	int TargetTexelLocation = Y * SizeX + X;

	float3 WorldNormal = make_float3(WorldNormalMap[TargetTexelLocation]);

	if (length(WorldNormal) < 0.01 || EmissiveMap[TargetTexelLocation].w != 1.0f)
		return;

	OutTaskBuffer->Buffer[atomicAggInc(&OutTaskBuffer->Size)] = TargetTexelLocation;
}

__global__ void RadiosityIncomingRadianceKernel(
	const float4* WorldPositionMap,
	const float4* WorldNormalMap,
	const float4* ReflectanceMap,
	const int SizeX,
	const int SizeY,
	const int RadiosityPassIndex,
	const int SurfaceCacheIndex,
	const int GridSpacing,
	const TaskBuffer* InTaskBuffer)
{
	if (blockIdx.x >= InTaskBuffer->Size)
		return;

	int TargetTexelLocation = InTaskBuffer->Buffer[blockIdx.x];
	int TexelX = TargetTexelLocation % SizeX;
	int TexelY = TargetTexelLocation / SizeX;
	float TexelU = (TexelX + 0.5f) / SizeX;
	float TexelV = (TexelY + 0.5f) / SizeY;

	if (threadIdx.x == 0 && threadIdx.y == 0)
		atomicAdd(&MappedTexelCounter, 1);

	float3 WorldPosition = make_float3(WorldPositionMap[TargetTexelLocation]);
	float3 WorldNormal = make_float3(WorldNormalMap[TargetTexelLocation]);

	float3 Tangent1, Tangent2;

	Tangent1 = cross(WorldNormal, make_float3(0, 0, 1));
	Tangent1 = length(Tangent1) < 0.1 ? cross(WorldNormal, make_float3(0, 1, 0)) : Tangent1;
	Tangent1 = normalize(Tangent1);
	Tangent2 = normalize(cross(Tangent1, WorldNormal));

	int X = blockIdx.x * blockDim.x + threadIdx.x;
	int Y = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = Y * SizeX + X;

#if USE_JITTERED_SAMPLING
	curandState randState;
#if USE_CORRELATED_SAMPLING
	curand_init(1, 0, 0, &randState);
#else
	curand_init(SizeX * SizeY * RadiosityPassIndex + threadId, 0, 0, &randState);
#endif
#endif

	__shared__ float3 GatheredRadiance[BLOCK_DIM_X][BLOCK_DIM_Y];
	__shared__ float InversedAverageDistance[BLOCK_DIM_X][BLOCK_DIM_Y];
	__shared__ int NumTotalHits[BLOCK_DIM_X][BLOCK_DIM_Y];
	__shared__ int NumBackfaceHits[BLOCK_DIM_X][BLOCK_DIM_Y];

	GatheredRadiance[threadIdx.x][threadIdx.y] = make_float3(0);
	InversedAverageDistance[threadIdx.x][threadIdx.y] = 0;
	NumBackfaceHits[threadIdx.x][threadIdx.y] = 0;
	NumTotalHits[threadIdx.x][threadIdx.y] = 0;

	const int NumRadiositySamples[] =
	{
		NumRadiositySampleFirstPass,
		NumRadiositySampleFirstPass * 0.5f,
		NumRadiositySampleFirstPass * 0.75f,
		NumRadiositySampleFirstPass * 0.75f,
		NumRadiositySampleFirstPass * 0.5f * 0.5f,
		NumRadiositySampleFirstPass * 0.5f * 0.5f * 0.75f,
		NumRadiositySampleFirstPass * 0.5f * 0.5f * 0.75f * 0.75f
	};

	const int NumThetaSteps = NumRadiositySamples[min(RadiosityPassIndex, 4)];
	const int NumPhiSteps = NumThetaSteps;
	const int NumSamples = NumThetaSteps * NumPhiSteps;

	for (int SampleIndexTheta = threadIdx.x; SampleIndexTheta < NumThetaSteps; SampleIndexTheta += BLOCK_DIM_X)
	{
		for (int SampleIndexPhi = threadIdx.y; SampleIndexPhi < NumPhiSteps; SampleIndexPhi += BLOCK_DIM_Y)
		{
			float RandA = 0.5f;
			float RandB = 0.5f;

		#if USE_JITTERED_SAMPLING
			RandA = curand_uniform(&randState);
			RandB = curand_uniform(&randState);
		#endif

			float U = 1.0f * (SampleIndexTheta + RandA) / NumThetaSteps;
			float Phi = 2.f * 3.1415926f * (SampleIndexPhi + RandB) / NumPhiSteps;
			float3 RayInLocalSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
		#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
			float2 StratifiedPoint2D = make_float2((SampleIndexTheta + RandA) / NumThetaSteps, (SampleIndexPhi + RandB) / NumPhiSteps);
			float2 ConcentricMappedPoint = ConcentricSampleDisk(StratifiedPoint2D);
			RayInLocalSpace = LiftPoint2DToHemisphere(ConcentricMappedPoint);
		#endif
			float3 RayInWorldSpace = normalize(Tangent1 * RayInLocalSpace.x + Tangent2 * RayInLocalSpace.y + WorldNormal * RayInLocalSpace.z);
			float3 RayOrigin = WorldPosition + WorldNormal * 0.5f;
			//RayOrigin += TexelRadius * Tangent1 * (0.5f - RandC) * 0.5f;
			//RayOrigin += TexelRadius * Tangent2 * (0.5f - RandD) * 0.5f;

			HitInfo OutHitInfo;

			rtTrace(
				OutHitInfo,
				make_float4(RayOrigin, 0.01),
				make_float4(RayInWorldSpace, 1e20), false);

			if (OutHitInfo.TriangleIndex == -1)
			{
				if (RadiosityPassIndex == 0)
				{
					if (SampleSkyLightRadiance(RayInWorldSpace).x >= 0)
					{
					#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
						float3 radiance = SampleSkyLightRadiance(RayInWorldSpace);
						if (getLuminance(radiance / RayInLocalSpace.z) > GPUSamplingGlobalParameters.FireflyClampingThreshold)
							radiance = radiance / getLuminance(radiance / RayInLocalSpace.z) * GPUSamplingGlobalParameters.FireflyClampingThreshold;
						GatheredRadiance[threadIdx.x][threadIdx.y] += radiance;
					#else
						GatheredRadiance[threadIdx.x][threadIdx.y] += SampleSkyLightRadiance(RayInWorldSpace) * make_float3(RayInLocalSpace.z);
					#endif
					}
				}
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
				float4 RadiantExitance = RadiositySurfaceCaches[HitSurfaceCacheIndex].cudaRadiositySurfaceCacheUnderlyingBuffer[(RadiosityPassIndex + 1) % 2][FinalY * RadiositySurfaceCaches[HitSurfaceCacheIndex].SizeX + FinalX];

				if (dot(OutHitInfo.TriangleNormalUnnormalized, RayInWorldSpace) < 0 || RadiantExitance.w == 1.0f)
				{
				#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
					float3 radiance = make_float3(RadiantExitance);
					if (getLuminance(radiance / RayInLocalSpace.z) > GPUSamplingGlobalParameters.FireflyClampingThreshold)
						radiance = radiance / getLuminance(radiance / RayInLocalSpace.z) * GPUSamplingGlobalParameters.FireflyClampingThreshold;
					GatheredRadiance[threadIdx.x][threadIdx.y] += radiance;
				#else
					GatheredRadiance[threadIdx.x][threadIdx.y] += make_float3(RayInLocalSpace.z * RadiantExitance);
				#endif
				}
				else
				{
					NumBackfaceHits[threadIdx.x][threadIdx.y]++;
				}
				if (SampleIndexTheta >= 2 && SampleIndexTheta < NumThetaSteps - 2)
					InversedAverageDistance[threadIdx.x][threadIdx.y] += 1.0f / OutHitInfo.HitDistance;
			}
		}
	}

#if USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING
	GatheredRadiance[threadIdx.x][threadIdx.y] *= 3.1415926f / NumSamples;
#else
	GatheredRadiance[threadIdx.x][threadIdx.y] *= 2 * 3.1415926f / NumSamples;
#endif

	if (RadiosityPassIndex == 0)
	{
		// SkyLights
		//for (int index = threadIdx.y * blockDim.x + threadIdx.x; index < 16 + 16; index += blockDim.x * blockDim.y)
		//{
		//	if (index < 16)
		//	{
		//		int sampleIndex = tex1Dfetch(SkyLightUpperHemisphereImportantDirectionsTexture, index);
		//		const int sampleTheta = sampleIndex / SkyLightCubemapNumPhiSteps;
		//		const int samplePhi = sampleIndex % SkyLightCubemapNumPhiSteps;
		//		float RandA = 0.5f;
		//		float RandB = 0.5f;
		//	#if USE_JITTERED_SAMPLING
		//		RandA = curand_uniform(&randState);
		//		RandB = curand_uniform(&randState);
		//	#endif
		//		float U = 1.0f * (sampleTheta + RandA) / SkyLightCubemapNumThetaSteps;
		//		float Phi = 2.f * 3.1415926f * (samplePhi + RandB) / (SkyLightCubemapNumPhiSteps);
		//		float3 RayOrigin = WorldPosition + WorldNormal * 0.1f;
		//		float3 RayInWorldSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
		//		float3 RayInLocalSpace =  normalize(make_float3(dot(RayInWorldSpace, Tangent1), dot(RayInWorldSpace, Tangent2), dot(RayInWorldSpace, WorldNormal)));
		//		if (RayInLocalSpace.z > 0)
		//		{
		//			HitInfo OutHitInfo;
		//			rtTrace(OutHitInfo, make_float4(RayOrigin, 0), make_float4(RayInWorldSpace, 1e20), true);
		//			if (OutHitInfo.TriangleIndex == -1)
		//			{
		//				const float SkylightCubemapResolution = 2.0f * SkyLightCubemapNumThetaSteps * SkyLightCubemapNumPhiSteps;
		//				const float PDF = SkylightCubemapResolution / (4 * 3.1415926f);
		//				GatheredRadiance[threadIdx.x][threadIdx.y] += make_float3(tex1Dfetch(SkyLightUpperHemisphereImportantColorTexture, index)) * RayInLocalSpace.z / PDF;
		//			}
		//		}
		//	}
		//	else
		//	{
		//		int sampleIndex = tex1Dfetch(SkyLightLowerHemisphereImportantDirectionsTexture, index - 16);
		//		const int sampleTheta = sampleIndex / SkyLightCubemapNumPhiSteps;
		//		const int samplePhi = sampleIndex % SkyLightCubemapNumPhiSteps;
		//		float RandA = 0.5f;
		//		float RandB = 0.5f;
		//	#if USE_JITTERED_SAMPLING
		//		RandA = curand_uniform(&randState);
		//		RandB = curand_uniform(&randState);
		//	#endif
		//		float U = 1.0f * (sampleTheta + RandA) / SkyLightCubemapNumThetaSteps;
		//		float Phi = 2.f * 3.1415926f * (samplePhi + RandB) / (SkyLightCubemapNumPhiSteps);
		//		float3 RayOrigin = WorldPosition + WorldNormal * 0.1f;
		//		float3 RayInWorldSpace = make_float3(sqrtf(1 - U * U) * cos(Phi), sqrtf(1 - U * U) * sin(Phi), U);
		//		float3 RayInLocalSpace =  normalize(make_float3(dot(RayInWorldSpace, Tangent1), dot(RayInWorldSpace, Tangent2), dot(RayInWorldSpace, WorldNormal)));
		//		if (RayInLocalSpace.z > 0)
		//		{
		//			HitInfo OutHitInfo;
		//			rtTrace(OutHitInfo, make_float4(RayOrigin, 0), make_float4(RayInWorldSpace, 1e20), true);
		//			if (OutHitInfo.TriangleIndex == -1)
		//			{
		//				const float SkylightCubemapResolution = 2.0f * SkyLightCubemapNumThetaSteps * SkyLightCubemapNumPhiSteps;
		//				const float PDF = SkylightCubemapResolution / (4 * 3.1415926f);
		//				GatheredRadiance[threadIdx.x][threadIdx.y] += make_float3(tex1Dfetch(SkyLightLowerHemisphereImportantColorTexture, index - 16)) * RayInLocalSpace.z / PDF;
		//			}
		//		}
		//	}
		//}

		// DirectionalLights
		for (int index = threadIdx.y * blockDim.x + threadIdx.x; index < NumDirectionalLights; index += blockDim.x * blockDim.y)
		{
			float3 RayInWorldSpace = normalize(-DirectionalLights[index].Direction);

			float3 Normal = WorldNormal;

			if (ReflectanceMap[TargetTexelLocation].w == 1.0f && dot(RayInWorldSpace, Normal) < 0.0f)
				Normal = -Normal;

			float3 RayOrigin = WorldPosition + Normal * 0.5f;

			HitInfo OutHitInfo;

			rtTrace(
				OutHitInfo,
				make_float4(RayOrigin, 0.01),
				make_float4(RayInWorldSpace, 1e20), true);

			if (OutHitInfo.TriangleIndex == -1)
			{
				GatheredRadiance[threadIdx.x][threadIdx.y] += DirectionalLights[index].Color * make_float3(max(dot(RayInWorldSpace, Normal), 0.0f));
			}
		}

		// PointLights
		for (int index = threadIdx.y * blockDim.x + threadIdx.x; index < NumPointLights; index += blockDim.x * blockDim.y)
		{
			float3 LightPosition = PointLights[index].WorldPosition;
			float Distance = length(WorldPosition - LightPosition);
			if (Distance < PointLights[index].Radius)
			{
				float3 RayOrigin = WorldPosition + WorldNormal * 0.5f;
				float3 RayInWorldSpace = normalize(LightPosition - WorldPosition);

				HitInfo OutHitInfo;

				rtTrace(
					OutHitInfo,
					make_float4(RayOrigin, 0.01),
					make_float4(RayInWorldSpace, Distance - 0.00001f), true);

				if (OutHitInfo.TriangleIndex == -1)
				{
					GatheredRadiance[threadIdx.x][threadIdx.y] += PointLights[index].Color * make_float3(max(dot(normalize(LightPosition - WorldPosition), WorldNormal), 0.0f)) / (Distance * Distance + 1.0f);
				}
			}
		}

		// SpotLights
		for (int index = threadIdx.y * blockDim.x + threadIdx.x; index < NumSpotLights; index += blockDim.x * blockDim.y)
		{
			float3 LightPosition = SpotLights[index].WorldPosition;
			float Distance = length(WorldPosition - LightPosition);
			if (Distance < SpotLights[index].Radius)
				if (dot(normalize(WorldPosition - LightPosition), SpotLights[index].Direction) > SpotLights[index].CosOuterConeAngle)
				{
					float3 RayOrigin = WorldPosition + WorldNormal * 0.5f;
					float3 RayInWorldSpace = normalize(LightPosition - WorldPosition);

					HitInfo OutHitInfo;

					rtTrace(
						OutHitInfo,
						make_float4(RayOrigin, 0.01),
						make_float4(RayInWorldSpace, Distance - 0.00001f), true);

					if (OutHitInfo.TriangleIndex == -1)
					{
						float SpotAttenuation = clampf(
							(dot(normalize(WorldPosition - LightPosition), SpotLights[index].Direction) - SpotLights[index].CosOuterConeAngle) / (SpotLights[index].CosInnerConeAngle - SpotLights[index].CosOuterConeAngle)
							, 0.0f, 1.0f);
						SpotAttenuation *= SpotAttenuation;
						GatheredRadiance[threadIdx.x][threadIdx.y] += SpotLights[index].Color * make_float3(max(dot(normalize(LightPosition - WorldPosition), WorldNormal), 0.0f)) / (Distance * Distance + 1.0f) * SpotAttenuation;
					}
				}
		}
	}

	for (int stride = BLOCK_DIM_X >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();

		if (threadIdx.x < stride)
		{
			GatheredRadiance[threadIdx.x][threadIdx.y] += GatheredRadiance[threadIdx.x + stride][threadIdx.y];
			NumTotalHits[threadIdx.x][threadIdx.y] += NumTotalHits[threadIdx.x + stride][threadIdx.y];
			NumBackfaceHits[threadIdx.x][threadIdx.y] += NumBackfaceHits[threadIdx.x + stride][threadIdx.y];
			InversedAverageDistance[threadIdx.x][threadIdx.y] += InversedAverageDistance[threadIdx.x + stride][threadIdx.y];
		}

		__syncthreads();

		if (threadIdx.y < stride)
		{
			GatheredRadiance[threadIdx.x][threadIdx.y] += GatheredRadiance[threadIdx.x][threadIdx.y + stride];
			NumTotalHits[threadIdx.x][threadIdx.y] += NumTotalHits[threadIdx.x][threadIdx.y + stride];
			NumBackfaceHits[threadIdx.x][threadIdx.y] += NumBackfaceHits[threadIdx.x][threadIdx.y + stride];
			InversedAverageDistance[threadIdx.x][threadIdx.y] += InversedAverageDistance[threadIdx.x][threadIdx.y + stride];
		}
	}

	float3 Irradiance = GatheredRadiance[0][0];

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		float bNotBackface = (NumTotalHits[0][0] < NumSamples * 0.4f || NumBackfaceHits[0][0] / NumTotalHits[0][0] < 0.1f) ? 1.0f : 0.0f;
		float AverageDistance = (NumThetaSteps - 4) * NumPhiSteps / InversedAverageDistance[0][0];
		RadiositySurfaceCaches[SurfaceCacheIndex].cudaRadiositySurfaceCacheUnderlyingBuffer[RadiosityPassIndex % 2][TargetTexelLocation] = make_float4(Irradiance, sqrtf(AverageDistance) * bNotBackface);
	}
}

__global__ void RadiosityApplyReflectanceKernel(
	const float4* ReflectanceMap,
	const float4* EmissiveMap,
	const int SizeX,
	const int SizeY,
	const int RadiosityPassIndex,
	const int SurfaceCacheIndex)
{
	int TexelX = blockIdx.x * blockDim.x + threadIdx.x;
	int TexelY = blockIdx.y * blockDim.y + threadIdx.y;

	int TargetTexelLocation = TexelY * SizeX + TexelX;

	if (TexelX >= SizeX || TexelY >= SizeY)
		return;

	float4& TexelValue = RadiositySurfaceCaches[SurfaceCacheIndex].cudaRadiositySurfaceCacheUnderlyingBuffer[RadiosityPassIndex % 2][TargetTexelLocation];

	float DiffuseBRDF = 1.0f / 3.1415926f;
	float3 Irradiance = make_float3(TexelValue);
	float4 Reflectance = ReflectanceMap[TargetTexelLocation];
	float3 Radiosity = Irradiance * DiffuseBRDF * make_float3(Reflectance) + (RadiosityPassIndex == 0 ? make_float3(EmissiveMap[TargetTexelLocation]) : make_float3(0.0f));

	if (Reflectance.w == 1.0f)
		Radiosity /= 2.0f;

	TexelValue = make_float4(Radiosity, Reflectance.w);

	RadiositySurfaceCaches[SurfaceCacheIndex].cudaFinalLightingStagingBuffer[TargetTexelLocation] += TexelValue;
}

__device__ void TryGetColorAndWorldPosition(
	const float4* WorldPositionMap,
	const int SizeX,
	const int SizeY,
	int RadiosityPassIndex, int SurfaceCacheIndex, int X, int Y, float4& OutColor, float3& OutWorldPosition)
{
	if (X >= 0 && X < SizeX && Y >= 0 && Y < SizeY)
	{
		int TargetTexelLocation = Y * SizeX + X;
		OutColor = RadiositySurfaceCaches[SurfaceCacheIndex].cudaRadiositySurfaceCacheUnderlyingBuffer[RadiosityPassIndex % 2][TargetTexelLocation];
		float u = (X + 0.5f) / SizeX;
		float v = (Y + 0.5f) / SizeY;
		OutWorldPosition = make_float3(WorldPositionMap[TargetTexelLocation]);
	}
	else {
		OutColor = make_float4(0);
		OutWorldPosition = make_float3(0);
	}
}

__device__ void AddTexelTask(
	int X, int Y,
	const int SizeX,
	const int SizeY, TaskBuffer* OutTaskBuffer)
{
	OutTaskBuffer->Buffer[atomicAggInc(&OutTaskBuffer->Size)] = Y * SizeX + X;
}

__global__ void RadiosityTryInterpolationKernel(
	const float4* WorldPositionMap,
	const float4* WorldNormalMap,
	const float4* EmissiveMap,
	const int SizeX,
	const int SizeY,
	int RadiosityPassIndex, int SurfaceCacheIndex, TaskBuffer* OutTaskBuffer, int GridSpacing)
{
	float4* const& IrradianceCache = RadiositySurfaceCaches[SurfaceCacheIndex].cudaRadiositySurfaceCacheUnderlyingBuffer[RadiosityPassIndex % 2];
	int X, Y;
	// First half
	X = (blockIdx.x * blockDim.x + threadIdx.x) * GridSpacing + GridSpacing / 2;
	Y = (blockIdx.y * blockDim.y + threadIdx.y) * GridSpacing;
	if (X < SizeX && Y < SizeY)
	{
		int TargetTexelLocation = Y * SizeX + X;

		float3 WorldNormal = make_float3(WorldNormalMap[TargetTexelLocation]);

		if (length(WorldNormal) >= 0.01 && EmissiveMap[TargetTexelLocation].w == 1.0f)
		{
			int BasePoint1 = X - GridSpacing / 2;
			int BasePoint2 = X + GridSpacing / 2;
			float4 BaseColor1, BaseColor2;
			float3 BasePointWorldPosition1, BasePointWorldPosition2;
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, BasePoint1, Y, BaseColor1, BasePointWorldPosition1);
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, BasePoint2, Y, BaseColor2, BasePointWorldPosition2);
			if (
				IRRADIANCE_CACHING_FORCE_NO_INTERPOLATION
				|| BaseColor1.w == 0 || BaseColor2.w == 0
			#if IRRADIANCE_CACHING_USE_DISTANCE
				|| length(BasePointWorldPosition1 - BasePointWorldPosition2) > fminf(BaseColor1.w, BaseColor2.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
			#endif
				)
			{
				AddTexelTask(X, Y, SizeX, SizeY, OutTaskBuffer);
			}
			else
			{
				IrradianceCache[TargetTexelLocation] = (BaseColor1 + BaseColor2) / 2;
			}
		}
	}

	// Second half
	X = (blockIdx.x * blockDim.x + threadIdx.x) * GridSpacing;
	Y = (blockIdx.y * blockDim.y + threadIdx.y) * GridSpacing + GridSpacing / 2;
	if (X < SizeX && Y < SizeY)
	{
		int TargetTexelLocation = Y * SizeX + X;

		float3 WorldNormal = make_float3(WorldNormalMap[TargetTexelLocation]);

		if (length(WorldNormal) >= 0.01 && EmissiveMap[TargetTexelLocation].w == 1.0f)
		{
			int BasePoint1 = Y - GridSpacing / 2;
			int BasePoint2 = Y + GridSpacing / 2;
			float4 BaseColor1, BaseColor2;
			float3 BasePointWorldPosition1, BasePointWorldPosition2;
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, X, BasePoint1, BaseColor1, BasePointWorldPosition1);
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, X, BasePoint2, BaseColor2, BasePointWorldPosition2);
			if (
				IRRADIANCE_CACHING_FORCE_NO_INTERPOLATION
				|| BaseColor1.w == 0 || BaseColor2.w == 0
			#if IRRADIANCE_CACHING_USE_DISTANCE
				|| length(BasePointWorldPosition1 - BasePointWorldPosition2) > fminf(BaseColor1.w, BaseColor2.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
			#endif
				)
			{
				AddTexelTask(X, Y, SizeX, SizeY, OutTaskBuffer);
			}
			else
			{
				IrradianceCache[TargetTexelLocation] = (BaseColor1 + BaseColor2) / 2;
			}
		}
	}

	// Center
	X = (blockIdx.x * blockDim.x + threadIdx.x) * GridSpacing + GridSpacing / 2;
	Y = (blockIdx.y * blockDim.y + threadIdx.y) * GridSpacing + GridSpacing / 2;
	if (X < SizeX && Y < SizeY)
	{
		int TargetTexelLocation = Y * SizeX + X;

		float3 WorldNormal = make_float3(WorldNormalMap[TargetTexelLocation]);

		if (length(WorldNormal) >= 0.01 && EmissiveMap[TargetTexelLocation].w == 1.0f)
		{
			float4 BaseColor1, BaseColor2, BaseColor3, BaseColor4;
			float3 BasePointWorldPosition1, BasePointWorldPosition2, BasePointWorldPosition3, BasePointWorldPosition4;
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, X - GridSpacing / 2, Y - GridSpacing / 2, BaseColor1, BasePointWorldPosition1);
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, X + GridSpacing / 2, Y - GridSpacing / 2, BaseColor2, BasePointWorldPosition2);
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, X - GridSpacing / 2, Y + GridSpacing / 2, BaseColor3, BasePointWorldPosition3);
			TryGetColorAndWorldPosition(WorldPositionMap, SizeX, SizeY, RadiosityPassIndex, SurfaceCacheIndex, X + GridSpacing / 2, Y + GridSpacing / 2, BaseColor4, BasePointWorldPosition4);

			float RelativeError1 = length(make_float3(BaseColor1 - BaseColor4)) / length(make_float3(fminf(BaseColor1, BaseColor4)));
			float RelativeError2 = length(make_float3(BaseColor2 - BaseColor3)) / length(make_float3(fminf(BaseColor2, BaseColor3)));
			if (
				IRRADIANCE_CACHING_FORCE_NO_INTERPOLATION
				|| BaseColor1.w == 0 || BaseColor2.w == 0 || BaseColor3.w == 0 || BaseColor4.w == 0
			#if IRRADIANCE_CACHING_USE_DISTANCE
				|| length(BasePointWorldPosition1 - BasePointWorldPosition2) > fminf(BaseColor1.w, BaseColor2.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
				|| length(BasePointWorldPosition1 - BasePointWorldPosition3) > fminf(BaseColor1.w, BaseColor3.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
				|| length(BasePointWorldPosition1 - BasePointWorldPosition4) > fminf(BaseColor1.w, BaseColor4.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
				|| length(BasePointWorldPosition2 - BasePointWorldPosition3) > fminf(BaseColor2.w, BaseColor3.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
				|| length(BasePointWorldPosition2 - BasePointWorldPosition4) > fminf(BaseColor2.w, BaseColor4.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
				|| length(BasePointWorldPosition3 - BasePointWorldPosition4) > fminf(BaseColor3.w, BaseColor4.w) * IRRADIANCE_CACHING_DISTANCE_SCALE
			#endif
				)
			{
				AddTexelTask(X, Y, SizeX, SizeY, OutTaskBuffer);
			}
			else
			{
				IrradianceCache[TargetTexelLocation] = (BaseColor1 + BaseColor2 + BaseColor3 + BaseColor4) / 4;
			}
		}
	}
}

#include "../ProgressReport.h"

__host__ void rtLaunchRadiosityLoop(int NumBounces, int NumSamplesFirstPass)
{
	int taskCount = 0;
	TaskBuffer* taskBuffer;
	cudaCheck(cudaMalloc(&taskBuffer, sizeof(TaskBuffer)));

	CreateSurfaceCacheTrasientDataOnGPU();

	GPULightmass::LOG("Total surface cache video memory data: %.2fMB", GetTotalVideoMemoryUsageDuringRadiosity() / 1024.0f / 1024.0f);

	std::vector<SurfaceCacheGPUDataPointers> Pointers = GenerateSurfaceCacheGPUPointers();
	SurfaceCacheGPUDataPointers* cudaRadiositySurfaceCaches = nullptr;
	cudaCheck(cudaMalloc(&cudaRadiositySurfaceCaches, Pointers.size() * sizeof(SurfaceCacheGPUDataPointers)));
	cudaCheck(cudaMemcpy(cudaRadiositySurfaceCaches, Pointers.data(), Pointers.size() * sizeof(SurfaceCacheGPUDataPointers), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpyToSymbol(RadiositySurfaceCaches, &cudaRadiositySurfaceCaches, 1 * sizeof(SurfaceCacheGPUDataPointers*)));

	cudaCheck(cudaMemcpyToSymbol(NumRadiositySampleFirstPass, &NumSamplesFirstPass, 1 * sizeof(int)));

	for (int RadiosityPassIndex = 0; RadiosityPassIndex < NumBounces; RadiosityPassIndex++)
	{
		GPULightmass::LOG("RadiosityPass %d", RadiosityPassIndex);

		for (int SurfaceCacheIndex = 0; SurfaceCacheIndex < SurfaceCaches.size(); SurfaceCacheIndex++)
		{
			ReportProgress(
				"Radiosity cache " + std::to_string(SurfaceCacheIndex + 1) + "/" + std::to_string(SurfaceCaches.size()),
				(int)(SurfaceCacheIndex * 100.0f / SurfaceCaches.size()),
				"Radiosity pass " + std::to_string(RadiosityPassIndex + 1) + "/" + std::to_string(NumBounces),
				(int)(RadiosityPassIndex * 100.0f / NumBounces)
			);

			if (SurfaceCaches[SurfaceCacheIndex].SizeX == 0)
				continue;

			cudaCheck(cudaMemset(taskBuffer, 0, 1 * sizeof(int)));

			LaunchSizeX = SurfaceCaches[SurfaceCacheIndex].SizeX;
			LaunchSizeY = SurfaceCaches[SurfaceCacheIndex].SizeY;

			UploadSurfaceCacheSampleDataToGPU(SurfaceCacheIndex);

			{
				dim3 blockDim(IRRADIANCE_CACHING_BASE_GRID_SPACING, IRRADIANCE_CACHING_BASE_GRID_SPACING);
				dim3 gridDim(
					divideAndRoundup(LaunchSizeX, IRRADIANCE_CACHING_BASE_GRID_SPACING),
					divideAndRoundup(LaunchSizeY, IRRADIANCE_CACHING_BASE_GRID_SPACING));
				RadiosityBaseGridGenerationKernel << < gridDim, blockDim >> > (
					SurfaceCaches[SurfaceCacheIndex].cudaWorldNormalMap,
					SurfaceCaches[SurfaceCacheIndex].cudaEmissiveMap,
					SurfaceCaches[SurfaceCacheIndex].SizeX,
					SurfaceCaches[SurfaceCacheIndex].SizeY,
					taskBuffer);
				cudaPostKernelLaunchCheck
			}

			cudaCheck(cudaMemcpy(&taskCount, taskBuffer, 1 * sizeof(int), cudaMemcpyDeviceToHost));

			for (int GridSpacing = IRRADIANCE_CACHING_BASE_GRID_SPACING; GridSpacing >= 2; GridSpacing /= 2)
			{
				if (taskCount > 0)
				{
					dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
					dim3 gridDim(taskCount, 1);
					RadiosityIncomingRadianceKernel << < gridDim, blockDim >> > (
						SurfaceCaches[SurfaceCacheIndex].cudaWorldPositionMap,
						SurfaceCaches[SurfaceCacheIndex].cudaWorldNormalMap,
						SurfaceCaches[SurfaceCacheIndex].cudaReflectanceMap,
						SurfaceCaches[SurfaceCacheIndex].SizeX,
						SurfaceCaches[SurfaceCacheIndex].SizeY,
						RadiosityPassIndex,
						SurfaceCacheIndex,
						GridSpacing,
						taskBuffer);
					cudaPostKernelLaunchCheck
				}

				cudaCheck(cudaMemset(taskBuffer, 0, 1 * sizeof(int)));

				{
					dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
					dim3 gridDim(
						divideAndRoundup(LaunchSizeX, BLOCK_DIM_X),
						divideAndRoundup(LaunchSizeY, BLOCK_DIM_Y));
					RadiosityTryInterpolationKernel << < gridDim, blockDim >> > (
						SurfaceCaches[SurfaceCacheIndex].cudaWorldPositionMap,
						SurfaceCaches[SurfaceCacheIndex].cudaWorldNormalMap,
						SurfaceCaches[SurfaceCacheIndex].cudaEmissiveMap,
						SurfaceCaches[SurfaceCacheIndex].SizeX,
						SurfaceCaches[SurfaceCacheIndex].SizeY,
						RadiosityPassIndex,
						SurfaceCacheIndex,
						taskBuffer, GridSpacing);
					cudaPostKernelLaunchCheck
				}

				cudaCheck(cudaMemcpy(&taskCount, taskBuffer, 1 * sizeof(int), cudaMemcpyDeviceToHost));
			}

			if (taskCount > 0)
			{
				dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
				dim3 gridDim(taskCount, 1);
				RadiosityIncomingRadianceKernel << < gridDim, blockDim >> > (
					SurfaceCaches[SurfaceCacheIndex].cudaWorldPositionMap,
					SurfaceCaches[SurfaceCacheIndex].cudaWorldNormalMap,
					SurfaceCaches[SurfaceCacheIndex].cudaReflectanceMap,
					SurfaceCaches[SurfaceCacheIndex].SizeX,
					SurfaceCaches[SurfaceCacheIndex].SizeY,
					RadiosityPassIndex,
					SurfaceCacheIndex,
					1,
					taskBuffer);
				cudaPostKernelLaunchCheck
			}

			{
				dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
				dim3 gridDim(
					divideAndRoundup(LaunchSizeX, BLOCK_DIM_X),
					divideAndRoundup(LaunchSizeY, BLOCK_DIM_Y));
				RadiosityApplyReflectanceKernel << < gridDim, blockDim >> > (
					SurfaceCaches[SurfaceCacheIndex].cudaReflectanceMap,
					SurfaceCaches[SurfaceCacheIndex].cudaEmissiveMap,
					SurfaceCaches[SurfaceCacheIndex].SizeX,
					SurfaceCaches[SurfaceCacheIndex].SizeY,
					RadiosityPassIndex, SurfaceCacheIndex);
				cudaPostKernelLaunchCheck
			}

			FreeSurfaceCacheSampleDataOnGPU(SurfaceCacheIndex);

		}

		ReportProgress(
			"Radiosity cache " + std::to_string(SurfaceCaches.size()) + "/" + std::to_string(SurfaceCaches.size()),
			100,
			"Radiosity pass " + std::to_string(RadiosityPassIndex + 1) + "/" + std::to_string(NumBounces),
			(int)(RadiosityPassIndex * 100.0f / NumBounces)
		);
	}

	ReportProgress(
		"Radiosity cache " + std::to_string(SurfaceCaches.size()) + "/" + std::to_string(SurfaceCaches.size()),
		100,
		"Radiosity pass " + std::to_string(NumBounces) + "/" + std::to_string(NumBounces),
		100,
		true
	);

	FreeSurfaceCacheTrasientData();

	cudaCheck(cudaFree(taskBuffer));
}
