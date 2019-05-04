__host__ void rtBindBVHData(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex,
	const unsigned int BVHSize,
	const unsigned int TriangleWoopSize,
	const unsigned int TriangleIndicesSize)
{
	GPULightmass::LOG("GPU BVH video memory size: %.2fMB, triangle payload size: %.2fMB",
		BVHSize * sizeof(float4) / 1024.0f / 1024.0f,
		(TriangleWoopSize * sizeof(float4) + TriangleIndicesSize * sizeof(int)) / 1024.0f / 1024.0f
	);

	cudaCheck(cudaMemcpyToSymbol(MappingFromTriangleAddressToIndex, &InMappingFromTriangleAddressToIndex, 1 * sizeof(InMappingFromTriangleAddressToIndex)));
	cudaCheck(cudaMemcpyToSymbol(TriangleWoopCoordinates, &InTriangleWoopCoordinates, 1 * sizeof(InTriangleWoopCoordinates)));
	cudaCheck(cudaMemcpyToSymbol(BVHTreeNodes, &InBVHTreeNodes, 1 * sizeof(InBVHTreeNodes)));

	cudaChannelFormatDesc channelDescFloat4 = cudaCreateChannelDesc<float4>();
	cudaCheck(cudaBindTexture(NULL, &BVHTreeNodesTexture, InBVHTreeNodes, &channelDescFloat4, BVHSize * sizeof(float4)));
	cudaCheck(cudaBindTexture(NULL, &TriangleWoopCoordinatesTexture, InTriangleWoopCoordinates, &channelDescFloat4, TriangleWoopSize * sizeof(float4)));

	cudaChannelFormatDesc channelDescInt = cudaCreateChannelDesc<int>();
	cudaCheck(cudaBindTexture(NULL, &MappingFromTriangleAddressToIndexTexture, InMappingFromTriangleAddressToIndex, &channelDescInt, TriangleIndicesSize * sizeof(int)));
}

__host__ void rtBindSampleData(
	const float4* SampleWorldPositions,
	const float4* SampleWorldNormals,
	const float* TexelRadius,
	GPULightmass::GatheredLightSample* InOutLightmapData,
	const int InSizeX,
	const int InSizeY)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaCheck(cudaBindTexture(NULL, &SampleWorldPositionsTexture, SampleWorldPositions, &channelDesc, InSizeX * InSizeY * sizeof(float4)));
	cudaCheck(cudaBindTexture(NULL, &SampleWorldNormalsTexture, SampleWorldNormals, &channelDesc, InSizeX * InSizeY * sizeof(float4)));

	cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
	cudaCheck(cudaBindTexture(NULL, &TexelRadiusTexture, TexelRadius, &channelDescFloat, InSizeX * InSizeY * sizeof(float)));

	cudaCheck(cudaMemcpyToSymbol(OutLightmapData, &InOutLightmapData, 1 * sizeof(GPULightmass::GatheredLightSample*)));
	cudaCheck(cudaMemcpyToSymbol(BindedSizeX, &InSizeX, 1 * sizeof(int)));
	cudaCheck(cudaMemcpyToSymbol(BindedSizeY, &InSizeY, 1 * sizeof(int)));

	LaunchSizeX = InSizeX;
	LaunchSizeY = InSizeY;
}

__host__ void rtBindParameterizationData(
	int*    InTriangleMappingIndex,
	int*    InTriangleIndexBuffer,
	float2* InVertexTextureUVs,
	float2* InVertexTextureLightmapUVs,
	int		TriangleCount,
	int		VertexCount
)
{
	cudaCheck(cudaMemcpyToSymbol(TriangleIndexBuffer, &InTriangleIndexBuffer, 1 * sizeof(InTriangleIndexBuffer)));
	cudaCheck(cudaMemcpyToSymbol(TriangleMappingIndex, &InTriangleMappingIndex, 1 * sizeof(InTriangleMappingIndex)));
	cudaCheck(cudaMemcpyToSymbol(VertexTextureUVs, &InVertexTextureUVs, 1 * sizeof(InVertexTextureUVs)));
	cudaCheck(cudaMemcpyToSymbol(VertexTextureLightmapUVs, &InVertexTextureLightmapUVs, 1 * sizeof(InVertexTextureLightmapUVs)));
}

__host__ void rtBindMaskedCollisionMaps(
	int NumMaps,
	cudaTextureObject_t* InMaps
)
{
	cudaTextureObject_t* cudaMaps;
	cudaCheck(cudaMalloc(&cudaMaps, NumMaps * sizeof(cudaTextureObject_t*)));
	cudaCheck(cudaMemcpy(cudaMaps, InMaps, NumMaps * sizeof(cudaTextureObject_t*), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpyToSymbol(MaskedCollisionMaps, &cudaMaps, 1 * sizeof(cudaTextureObject_t*)));
}

__host__ void rtBindSkyCubemapData(
	const int NumThetaSteps,
	const int NumPhiSteps,
	const float4 UpperHemisphereCubemap[],
	const float4 LowerHemisphereCubemap[],
	const int UpperHemisphereImportantDirections[],
	const int LowerHemisphereImportantDirections[],
	const float4 InUpperHemisphereImportantColor[],
	const float4 InLowerHemisphereImportantColor[]
)
{
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaCheck(cudaBindTexture(NULL, &SkyLightUpperHemisphereTexture, UpperHemisphereCubemap, &channelDesc, NumThetaSteps * NumPhiSteps * sizeof(float4)));
		cudaCheck(cudaBindTexture(NULL, &SkyLightLowerHemisphereTexture, LowerHemisphereCubemap, &channelDesc, NumThetaSteps * NumPhiSteps * sizeof(float4)));
		cudaCheck(cudaBindTexture(NULL, &SkyLightUpperHemisphereImportantColorTexture, InUpperHemisphereImportantColor, &channelDesc, 16 * sizeof(float4)));
		cudaCheck(cudaBindTexture(NULL, &SkyLightLowerHemisphereImportantColorTexture, InLowerHemisphereImportantColor, &channelDesc, 16 * sizeof(float4)));
	}

	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
		cudaCheck(cudaBindTexture(NULL, &SkyLightUpperHemisphereImportantDirectionsTexture, UpperHemisphereImportantDirections, &channelDesc, 16 * sizeof(int)));
		cudaCheck(cudaBindTexture(NULL, &SkyLightLowerHemisphereImportantDirectionsTexture, LowerHemisphereImportantDirections, &channelDesc, 16 * sizeof(int)));
	}

	cudaCheck(cudaMemcpyToSymbol(SkyLightCubemapNumThetaSteps, &NumThetaSteps, 1 * sizeof(int)));
	cudaCheck(cudaMemcpyToSymbol(SkyLightCubemapNumPhiSteps, &NumPhiSteps, 1 * sizeof(int)));
}

__host__ void rtBindPunctualLights(
	const int InNumDirectionalLights,
	const GPULightmass::DirectionalLight* InDirectionalLights,
	const int InNumPointLights,
	const GPULightmass::PointLight* InPointLights,
	const int InNumSpotLights,
	const GPULightmass::SpotLight* InSpotLights
)
{
	cudaCheck(cudaMemcpyToSymbol(NumDirectionalLights, &InNumDirectionalLights, 1 * sizeof(int)));
	cudaCheck(cudaMemcpyToSymbol(DirectionalLights, &InDirectionalLights, 1 * sizeof(GPULightmass::DirectionalLight*)));
	cudaCheck(cudaMemcpyToSymbol(NumPointLights, &InNumPointLights, 1 * sizeof(int)));
	cudaCheck(cudaMemcpyToSymbol(PointLights, &InPointLights, 1 * sizeof(GPULightmass::PointLight*)));
	cudaCheck(cudaMemcpyToSymbol(NumSpotLights, &InNumSpotLights, 1 * sizeof(int)));
	cudaCheck(cudaMemcpyToSymbol(SpotLights, &InSpotLights, 1 * sizeof(GPULightmass::SpotLight*)));
}

__host__ void rtSetGlobalSamplingParameters(
	float FireflyClampingThreshold
)
{
	SamplingGlobalParameters Parameters;
	Parameters.FireflyClampingThreshold = FireflyClampingThreshold;

	cudaCheck(cudaMemcpyToSymbol(GPUSamplingGlobalParameters, &Parameters, 1 * sizeof(SamplingGlobalParameters)));
}

double LastRayTracingPerformance = 0.0f;
double OverallRayTracingPerformance = 0.0f;

float elapsedTime = 0.0f;
double accumulatedGPUTime = 0.0f;

__host__ double rtLaunchFinalGather(int NumSamples)
{
	static int NumFinishedTextureMappings = 0;
	int Counter = 0;
	MortonHash6* cudaMortonHashes;

	{
		AABB InitAABB;
		InitAABB.Low = make_float3(FLT_MAX);
		InitAABB.High = make_float3(-FLT_MAX);

		cudaCheck(cudaMemcpyToSymbol(WorldPositionAABB, &InitAABB, 1 * sizeof(AABB)));

		{
			const int Stride = 64;
			dim3 blockDim(Stride, 1);
			dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

			ComputeAABB << < gridDim, blockDim >> > ();
		}

		cudaCheck(cudaMalloc((void**)&cudaMortonHashes, LaunchSizeX * LaunchSizeY * sizeof(MortonHash6)));

		{
			const int Stride = 64;
			dim3 blockDim(Stride, 1);
			dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

			ComputeMortonHash << < gridDim, blockDim >> > (cudaMortonHashes);
		}

		cudaCheck(cudaMemcpyFromSymbol(&Counter, MappedTexelCounter, 1 * sizeof(int)));

		MortonHash6* ComputedHashes;
		cudaHostAlloc(&ComputedHashes, Counter * sizeof(MortonHash6), 0);

		cudaCheck(cudaMemcpy(ComputedHashes, cudaMortonHashes, Counter * sizeof(MortonHash6), cudaMemcpyDeviceToHost));

		qsort(ComputedHashes, Counter, sizeof(MortonHash6), compareMortonKey);

		cudaCheck(cudaMemcpy(cudaMortonHashes, ComputedHashes, Counter * sizeof(MortonHash6), cudaMemcpyHostToDevice));
	}

	double numTotalRay = 0.0f;

	SamplingParameters samplingParameters;

	samplingParameters.NumSampleTheta = NumSamples;
	samplingParameters.NumSamplePhi = samplingParameters.NumSampleTheta;
	samplingParameters.TotalSamplePerTexel = samplingParameters.NumSampleTheta * samplingParameters.NumSamplePhi;
	samplingParameters.NumSamplePerBucketTheta = samplingParameters.NumSampleTheta / NumBucketTheta;
	samplingParameters.NumSamplePerBucketPhi = samplingParameters.NumSamplePhi / NumBucketPhi;
	samplingParameters.TotalSamplePerBucket = samplingParameters.NumSamplePerBucketTheta * samplingParameters.NumSamplePerBucketPhi;
	samplingParameters.ImageBlockSize = MaxRayBufferSize / samplingParameters.TotalSamplePerTexel;

	cudaCheck(cudaMemcpyToSymbol(GPUSamplingParameters, &samplingParameters, sizeof(SamplingParameters)));

	float4*	cudaBucketRadiance;
	int*	cudaBucketSampleRejected;
	float*	cudaDownsampledBucketImportance;
	int*	cudaBucketRayStartOffsetInTexel;
	int*	cudaTexelToRayIDMap;

	cudaCheck(cudaMalloc(&cudaBucketRadiance, sizeof(float4) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));
	cudaCheck(cudaMalloc(&cudaBucketSampleRejected, sizeof(int) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));
	cudaCheck(cudaMalloc(&cudaDownsampledBucketImportance, sizeof(float) * samplingParameters.ImageBlockSize / ImportanceImageFilteringFactor * TotalBucketPerTexel));
	cudaCheck(cudaMalloc(&cudaBucketRayStartOffsetInTexel, sizeof(int) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));
	cudaCheck(cudaMalloc(&cudaTexelToRayIDMap, sizeof(int) * samplingParameters.ImageBlockSize));

	cudaCheck(cudaMemcpyToSymbol(BucketRadiance, &cudaBucketRadiance, 1 * sizeof(float4*)));
	cudaCheck(cudaMemcpyToSymbol(BucketSampleRejected, &cudaBucketSampleRejected, 1 * sizeof(int*)));
	cudaCheck(cudaMemcpyToSymbol(DownsampledBucketImportance, &cudaDownsampledBucketImportance, 1 * sizeof(float*)));
	cudaCheck(cudaMemcpyToSymbol(BucketRayStartOffsetInTexel, &cudaBucketRayStartOffsetInTexel, 1 * sizeof(int*)));
	cudaCheck(cudaMemcpyToSymbol(TexelToRayIDMap, &cudaTexelToRayIDMap, 1 * sizeof(int*)));

	void* cudaRayBuffer;
	void* cudaRayStartInfoBuffer;
	void* cudaRayHitResultBuffer;
	cudaCheck(cudaMalloc(&cudaRayBuffer, sizeof(Ray) * MaxRayBufferSize));
	cudaCheck(cudaMalloc(&cudaRayStartInfoBuffer, sizeof(RayStartInfo) * MaxRayBufferSize));
	cudaCheck(cudaMalloc(&cudaRayHitResultBuffer, sizeof(RayResult) * MaxRayBufferSize));
	cudaCheck(cudaMemcpyToSymbol(RayBuffer, &cudaRayBuffer, 1 * sizeof(Ray*)));
	cudaCheck(cudaMemcpyToSymbol(RayStartInfoBuffer, &cudaRayStartInfoBuffer, 1 * sizeof(RayStartInfo*)));
	cudaCheck(cudaMemcpyToSymbol(RayHitResultBuffer, &cudaRayHitResultBuffer, 1 * sizeof(RayResult*)));

	for (int i = 0; i < divideAndRoundup(Counter, samplingParameters.ImageBlockSize); i++)
	{
		ReportProgressTextureMapping(
			i,
			divideAndRoundup(Counter, samplingParameters.ImageBlockSize),
			elapsedTime,
			LastRayTracingPerformance,
			accumulatedGPUTime,
			OverallRayTracingPerformance
		);

		int rayCount = 0;

		cudaCheck(cudaMemcpyToSymbol(RayCount, &rayCount, 1 * sizeof(int)));
		cudaCheck(cudaMemcpyToSymbol(FinishedRayCount, &rayCount, 1 * sizeof(int)));

		cudaCheck(cudaMemset(cudaBucketRadiance, 0, sizeof(float4) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));
		cudaCheck(cudaMemset(cudaBucketSampleRejected, 0, sizeof(int) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));
		cudaCheck(cudaMemset(cudaBucketRayStartOffsetInTexel, 0, sizeof(int) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));

		cudaCheck(cudaMemset(cudaRayBuffer, 0, sizeof(Ray) * samplingParameters.ImageBlockSize * samplingParameters.TotalSamplePerTexel));
		{
			dim3 blockDim(32 * 2, 1);
			dim3 gridDim(divideAndRoundup(samplingParameters.ImageBlockSize * samplingParameters.TotalSamplePerTexel, blockDim.x), 1);

			BucketRayGenKernel << < gridDim, blockDim >> > (
				i * samplingParameters.ImageBlockSize,
				cudaMortonHashes
				);
			cudaPostKernelLaunchCheck
		}

		cudaCheck(cudaMemcpyFromSymbol(&rayCount, RayCount, 1 * sizeof(int)));

		numTotalRay += rayCount;

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(32 * 32, 1);

			rtTraceDynamicFetch << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(samplingParameters.ImageBlockSize * TotalBucketPerTexel, 1);
			BucketGatherKernel << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

#if USE_ADAPTIVE_SAMPLING

		numTotalRay += rayCount;
		cudaCheck(cudaMemset(cudaBucketRayStartOffsetInTexel, 0, sizeof(int) * samplingParameters.ImageBlockSize * TotalBucketPerTexel));

		{
			dim3 blockDim(TotalBucketPerTexel, 1);
			dim3 gridDim(divideAndRoundup(samplingParameters.ImageBlockSize, ImportanceImageFilteringFactor), 1);
			DownsampleImportanceKernel << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

		{
			dim3 blockDim(TotalBucketPerTexel, 1);
			dim3 gridDim(samplingParameters.ImageBlockSize, 1);
			ScatterImportanceAndCalculateSampleNumKernel << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

		rayCount = 0;
		cudaCheck(cudaMemcpyToSymbol(RayCount, &rayCount, 1 * sizeof(int)));
		cudaCheck(cudaMemcpyToSymbol(FinishedRayCount, &rayCount, 1 * sizeof(int)));

		cudaCheck(cudaMemset(cudaRayBuffer, 0, sizeof(Ray) * MaxRayBufferSize));
		{
			dim3 blockDim(32, 2);
			dim3 gridDim(samplingParameters.ImageBlockSize * TotalBucketPerTexel, 1);
			BucketAdaptiveRayGenKernel << < gridDim, blockDim >> > (
				i * samplingParameters.ImageBlockSize,
				cudaMortonHashes
				);
			cudaPostKernelLaunchCheck
		}

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(32 * 32, 1);

			rtTraceDynamicFetch << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}

		{
			dim3 blockDim(32, 2);
			dim3 gridDim(samplingParameters.ImageBlockSize * TotalBucketPerTexel, 1);
			BucketGatherKernel << < gridDim, blockDim >> > ();
			cudaPostKernelLaunchCheck
		}
#endif

		{
			dim3 blockDim(NumBucketTheta, NumBucketPhi);
			dim3 gridDim(samplingParameters.ImageBlockSize, 1);

			BucketShadingKernel << < gridDim, blockDim >> > (
				i * samplingParameters.ImageBlockSize,
				cudaMortonHashes
				);
			cudaPostKernelLaunchCheck
		}

	#if 0
		{
			dim3 blockDim(16, 32);
			dim3 gridDim(samplingParameters.ImageBlockSize, 1);

			SkylightImportanceSampleKernel <<< gridDim, blockDim >>> (
				i * samplingParameters.ImageBlockSize,
				cudaMortonHashes
				);
			cudaPostKernelLaunchCheck
		}
	#endif
	}

	cudaFree(cudaMortonHashes);

	cudaFree(cudaRayBuffer);
	cudaFree(cudaRayStartInfoBuffer);
	cudaFree(cudaRayHitResultBuffer);

	cudaFree(cudaBucketRadiance);
	cudaFree(cudaBucketSampleRejected);
	cudaFree(cudaDownsampledBucketImportance);
	cudaFree(cudaBucketRayStartOffsetInTexel);
	cudaFree(cudaTexelToRayIDMap);

	NumFinishedTextureMappings++;

	ReportProgressTextureMapping(
		divideAndRoundup(Counter, samplingParameters.ImageBlockSize) - 1,
		divideAndRoundup(Counter, samplingParameters.ImageBlockSize),
		elapsedTime,
		LastRayTracingPerformance,
		accumulatedGPUTime,
		OverallRayTracingPerformance
	);

	return numTotalRay;
}

__host__ float rtTimedLaunchRadiosity(int NumBounces, int NumSamplesFirstPass)
{
	float elapsedTime;
	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));

	int Counter = 0;
	cudaCheck(cudaMemcpyToSymbol(MappedTexelCounter, &Counter, 1 * sizeof(int)));

	cudaDeviceSynchronize();

	cudaProfilerStart();

	cudaCheck(cudaEventRecord(startEvent, 0));

	rtLaunchRadiosityLoop(NumBounces, NumSamplesFirstPass);

	cudaCheck(cudaEventRecord(stopEvent, 0));

	cudaCheck(cudaEventSynchronize(stopEvent));

	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	cudaProfilerStop();

	cudaDeviceSynchronize();

	cudaCheck(cudaMemcpyFromSymbol(&Counter, MappedTexelCounter, 1 * sizeof(int)));

	static float accumulatedGPUTime = 0;
	accumulatedGPUTime += elapsedTime;

	float MRaysPerSecond = (float)Counter * SampleCountOneDimension * SampleCountOneDimension * 4 / 4 / 1000000.0f / (elapsedTime / 1000.0f);

	GPULightmass::LOG("GPU accumulated time: %.3fs (Current: %.3fMS), %.2fMRays/s, %d texels", accumulatedGPUTime / 1000.0f, elapsedTime, MRaysPerSecond, Counter);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		GPULightmass::LOG("PostKernelLaunchError: %s", cudaGetErrorString(err));

	return elapsedTime;
}

__host__ float rtTimedLaunch(float& OutMRaysPerSecond, int NumSamples)
{
	cudaEvent_t startEvent, stopEvent;
	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));

	int Counter = 0;
	cudaCheck(cudaMemcpyToSymbol(MappedTexelCounter, &Counter, 1 * sizeof(int)));

	cudaDeviceSynchronize();

	cudaProfilerStart();

	cudaCheck(cudaEventRecord(startEvent, 0));

	double numTotalRay = rtLaunchFinalGather(NumSamples);

	cudaCheck(cudaEventRecord(stopEvent, 0));

	cudaCheck(cudaEventSynchronize(stopEvent));

	cudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	cudaProfilerStop();

	cudaCheck(cudaMemcpyFromSymbol(&Counter, MappedTexelCounter, 1 * sizeof(int)));

	static double totalRayTraced = 0.0;

	accumulatedGPUTime += elapsedTime;
	totalRayTraced += numTotalRay;

	OutMRaysPerSecond = totalRayTraced / 1000000.0 / (accumulatedGPUTime / 1000.0);
	const float CurrentMRaysPerSecond = numTotalRay / 1000000.0 / (elapsedTime / 1000.0);

	LastRayTracingPerformance = numTotalRay / 1000000.0 / (elapsedTime / 1000.0);
	OverallRayTracingPerformance = totalRayTraced / 1000000.0 / (accumulatedGPUTime / 1000.0);

	GPULightmass::LOG("GPU accumulated time: %.3lfs (Current: %.3fMS), %.2lfMRays/s (Current: %.2lfMRays/s), %d texels, %dx%d texture", accumulatedGPUTime / 1000.0, elapsedTime, OutMRaysPerSecond, CurrentMRaysPerSecond, Counter, LaunchSizeX, LaunchSizeY);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		GPULightmass::LOG("PostKernelLaunchError: %s", cudaGetErrorString(err));

	return elapsedTime;
}

__host__ void cudaGenerateSignedDistanceFieldVolumeData(Vec3f BoundingBoxMin, Vec3f BoundingBoxMax, Vec3i VolumeDimension, float* OutBuffer, int ZSliceIndex)
{
	dim3 block(8, 8, 8);
	int gridSizeX = (int)ceilf((float)VolumeDimension.x / 8);
	int gridSizeY = (int)ceilf((float)VolumeDimension.y / 8);
	dim3 grid(gridSizeX, gridSizeY, 1);

	const Vec3f VolumeSize = BoundingBoxMax - BoundingBoxMin;
	const Vec3f DistanceFieldVoxelSize = VolumeSize / Vec3f(VolumeDimension.x, VolumeDimension.y, VolumeDimension.z);

	GenerateSignedDistanceFieldVolumeDataKernel << < grid, block >> > (BoundingBoxMin, BoundingBoxMax, VolumeDimension, OutBuffer, ZSliceIndex);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		GPULightmass::LOG("PostKernelLaunchError: %s", cudaGetErrorString(err));

	cudaDeviceSynchronize();
}
