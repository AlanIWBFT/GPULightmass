#include "helper_math.h"

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
};

}

#include "HostFunc.h"
#include "rt/rtDebugFunc.h"

__host__ void rtBindBVHData(
	const float4* InBVHTreeNodes,
	const float4* InTriangleWoopCoordinates,
	const int* InMappingFromTriangleAddressToIndex,
	const unsigned int BVHSize,
	const unsigned int TriangleWoopSize,
	const unsigned int TriangleIndicesSize);

__host__ void rtBindSampleData(
	const float4* SampleWorldPositions,
	const float4* SampleWorldNormals,
	const float* TexelRadius,
	GPULightmass::GatheredLightSample* InOutLightmapData,
	const int InSizeX,
	const int InSizeY);

__host__ void rtBindParameterizationData(
	int*    InTriangleMaterialIndex,
	int*    InTriangleIndexBuffer,
	float2* InVertexTextureUVs,
	float2* InVertexTextureLightmapUVs,
	int		TriangleCount,
	int		VertexCount
);

__host__ void rtBindSkyCubemapData(
	const int NumThetaSteps,
	const int NumPhiSteps,
	const float4 UpperHemisphereCubemap[],
	const float4 LowerHemisphereCubemap[],
	const int UpperHemisphereImportantDirections[],
	const int LowerHemisphereImportantDirections[],
	const float4 InUpperHemisphereImportantColor[],
	const float4 InLowerHemisphereImportantColor[]
);

float rtTimedLaunch(float& OutMRaysPerSecond, int NumSamples);

float rtTimedLaunchRadiosity(int NumBounces, int NumSamplesFirstPass);

void cudaGenerateSignedDistanceFieldVolumeData(Vec3f BoundingBoxMin, Vec3f BoundingBoxMax, Vec3i VolumeDimension, float* OutBuffer, int ZSliceIndex);

void BindBVHData(
	std::vector<float4>& NodeData,
	std::vector<float4>& WoopifiedTriangles,
	std::vector<int>& TriangleIndices)
{
	float4* cudaBVHNodes = NULL;
	float4* cudaTriangleWoopCoordinates = NULL;
	int*    cudaMappingFromTriangleAddressToIndex = NULL;

	cudaCheck(cudaMalloc((void**)&cudaBVHNodes, NodeData.size() * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaBVHNodes, NodeData.data(), NodeData.size() * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTriangleWoopCoordinates, WoopifiedTriangles.size() * sizeof(float4)));
	cudaCheck(cudaMemcpy(cudaTriangleWoopCoordinates, WoopifiedTriangles.data(), WoopifiedTriangles.size() * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaMappingFromTriangleAddressToIndex, TriangleIndices.size() * sizeof(int)));
	cudaCheck(cudaMemcpy(cudaMappingFromTriangleAddressToIndex, TriangleIndices.data(), TriangleIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

	rtBindBVHData(cudaBVHNodes, cudaTriangleWoopCoordinates, cudaMappingFromTriangleAddressToIndex, NodeData.size(), WoopifiedTriangles.size(), TriangleIndices.size());
}

void BindParameterizationData(
	const int NumVertices,
	const int NumTriangles,
	const float2 VertexTextureUVBuffer[],
	const float2 VertexLightmapUVBuffer[],
	const int3 TriangleIndexBuffer[],
	const int TriangleTextureMappingIndex[])
{
	int*    cudaTriangleMappingIndexPtr = NULL;
	int*    cudaTriangleIndexBufferPtr = NULL;
	float2* cudaVertexTextureUVs = NULL;
	float2* cudaVertexTextureLightmapUVs = NULL;

	GPULightmass::LOG("GPU triangle parameterization data size: %.2lfMB",
		((double)NumTriangles * sizeof(int3) +
		(double)NumVertices * sizeof(float2) +
		(double)NumVertices * sizeof(float2) +
		(double)NumTriangles * sizeof(int))
		/ 1024.0 / 1024.0
	);

	cudaMalloc((void**)&cudaTriangleIndexBufferPtr, NumTriangles * sizeof(int3));
	cudaCheck(cudaMemcpy(cudaTriangleIndexBufferPtr, TriangleIndexBuffer, NumTriangles * sizeof(int3), cudaMemcpyHostToDevice));

	cudaMalloc((void**)&cudaVertexTextureUVs, NumVertices * sizeof(float2));
	cudaCheck(cudaMemcpy(cudaVertexTextureUVs, VertexTextureUVBuffer, NumVertices * sizeof(float2), cudaMemcpyHostToDevice));

	cudaMalloc((void**)&cudaVertexTextureLightmapUVs, NumVertices * sizeof(float2));
	cudaCheck(cudaMemcpy(cudaVertexTextureLightmapUVs, VertexLightmapUVBuffer, NumVertices * sizeof(float2), cudaMemcpyHostToDevice));

	cudaMalloc(&cudaTriangleMappingIndexPtr, NumTriangles * sizeof(int));
	cudaCheck(cudaMemcpy(cudaTriangleMappingIndexPtr, TriangleTextureMappingIndex, NumTriangles * sizeof(int), cudaMemcpyHostToDevice));

	rtBindParameterizationData(cudaTriangleMappingIndexPtr, cudaTriangleIndexBufferPtr, cudaVertexTextureUVs, cudaVertexTextureLightmapUVs, NumTriangles, NumVertices);
}

void LaunchRadiosityLoop(int NumBounces, int NumSamplesFirstPass)
{
	rtTimedLaunchRadiosity(NumBounces, NumSamplesFirstPass);
}

void BindSkyLightCubemap(
	const int NumThetaSteps,
	const int NumPhiSteps,
	const float4 InUpperHemisphereCubemap[],
	const float4 InLowerHemisphereCubemap[], 
	const int UpperHemisphereImportantDirections[],
	const int LowerHemisphereImportantDirections[],
	const float4 InUpperHemisphereImportantColor[],
	const float4 InLowerHemisphereImportantColor[]
)
{
	float4* cudaUpperHemisphereCubemap;
	float4* cudaLowerHemisphereCubemap;
	int* cudaUpperHemisphereImportantDirections;
	int* cudaLowerHemisphereImportantDirections;
	float4* cudaUpperHemisphereImportantColor;
	float4* cudaLowerHemisphereImportantColor;

	cudaCheck(cudaMalloc((void**)&cudaUpperHemisphereCubemap, NumThetaSteps * NumPhiSteps * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&cudaLowerHemisphereCubemap, NumThetaSteps * NumPhiSteps * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&cudaUpperHemisphereImportantDirections, 16 * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&cudaLowerHemisphereImportantDirections, 16 * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&cudaUpperHemisphereImportantColor, 16 * sizeof(float4)));
	cudaCheck(cudaMalloc((void**)&cudaLowerHemisphereImportantColor, 16 * sizeof(float4)));

	cudaCheck(cudaMemcpy(cudaUpperHemisphereCubemap, InUpperHemisphereCubemap, NumThetaSteps * NumPhiSteps * sizeof(float4), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaLowerHemisphereCubemap, InLowerHemisphereCubemap, NumThetaSteps * NumPhiSteps * sizeof(float4), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaUpperHemisphereImportantDirections, UpperHemisphereImportantDirections, 16 * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaLowerHemisphereImportantDirections, LowerHemisphereImportantDirections, 16 * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaUpperHemisphereImportantColor, InUpperHemisphereImportantColor, 16 * sizeof(float4), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaLowerHemisphereImportantColor, InLowerHemisphereImportantColor, 16 * sizeof(float4), cudaMemcpyHostToDevice));

	rtBindSkyCubemapData(NumThetaSteps, NumPhiSteps, cudaUpperHemisphereCubemap, cudaLowerHemisphereCubemap, cudaUpperHemisphereImportantDirections, cudaLowerHemisphereImportantDirections, cudaUpperHemisphereImportantColor, cudaLowerHemisphereImportantColor);
}

bool BindBVHDataFromFile(std::string fileName)
{
	FILE* fp;

	fp = fopen(fileName.c_str(), "rb");

	if (fp != nullptr)
	{
		GPULightmass::LOG("Loading cached BVH from %s", fileName.c_str());

		U32 gpuNodesSize;
		U32 gpuTriWoopSize;
		U32 gpuTriIndicesSize;
		U32 leafnodeCount;
		U32 triCount;

		fread(&gpuNodesSize, sizeof(gpuNodesSize), 1, fp);
		fread(&gpuTriWoopSize, sizeof(gpuTriWoopSize), 1, fp);
		fread(&gpuTriIndicesSize, sizeof(gpuTriIndicesSize), 1, fp);
		fread(&leafnodeCount, sizeof(leafnodeCount), 1, fp);
		fread(&triCount, sizeof(triCount), 1, fp);

		std::unique_ptr<Vec4i[]> gpuNodes { new Vec4i[gpuNodesSize]() };
		std::unique_ptr<Vec4i[]> gpuTriWoop { new Vec4i[gpuTriWoopSize]() };
		std::unique_ptr<int[]> gpuTriIndices { new int[gpuTriIndicesSize]() };

		fread(gpuNodes.get(), sizeof(float4), gpuNodesSize, fp);
		fread(gpuTriWoop.get(), sizeof(float4), gpuTriWoopSize, fp);
		fread(gpuTriIndices.get(), sizeof(int), gpuTriIndicesSize, fp);

		fclose(fp);

		float4* cudaBVHNodes = NULL;
		float4* cudaTriangleWoopCoordinates = NULL;
		int*    cudaMappingFromTriangleAddressToIndex = NULL;

		cudaCheck(cudaMalloc((void**)&cudaBVHNodes, gpuNodesSize * sizeof(float4)));
		cudaCheck(cudaMemcpy(cudaBVHNodes, gpuNodes.get(), gpuNodesSize * sizeof(float4), cudaMemcpyHostToDevice));

		cudaCheck(cudaMalloc((void**)&cudaTriangleWoopCoordinates, gpuTriWoopSize * sizeof(float4)));
		cudaCheck(cudaMemcpy(cudaTriangleWoopCoordinates, gpuTriWoop.get(), gpuTriWoopSize * sizeof(float4), cudaMemcpyHostToDevice));

		cudaCheck(cudaMalloc((void**)&cudaMappingFromTriangleAddressToIndex, gpuTriIndicesSize * sizeof(int)));
		cudaCheck(cudaMemcpy(cudaMappingFromTriangleAddressToIndex, gpuTriIndices.get(), gpuTriIndicesSize * sizeof(int), cudaMemcpyHostToDevice));

		rtBindBVHData(cudaBVHNodes, cudaTriangleWoopCoordinates, cudaMappingFromTriangleAddressToIndex, gpuNodesSize, gpuTriWoopSize, gpuTriIndicesSize);

		return true;
	}

	return false;
}

void SaveSampleDataToFile(const float4* SampleWorldPositions, const float4 * SampleWorldNormals, const float * TexelRadius, int SizeX, int SizeY, std::string fileName)
{
	FILE* fp;
	fp = fopen(fileName.c_str(), "wb");
	fwrite(&SizeX, sizeof(SizeX), 1, fp);
	fwrite(&SizeY, sizeof(SizeY), 1, fp);
	fwrite(SampleWorldPositions, sizeof(float4), SizeX * SizeY, fp);
	fwrite(SampleWorldNormals, sizeof(float4), SizeX * SizeY, fp);
	fwrite(TexelRadius, sizeof(float), SizeX * SizeY, fp);
	fclose(fp);
}

void CalculateLighting(
	const float4 * SampleWorldPositions,
	const float4 * SampleWorldNormals,
	const float* TexelRadius,
	GPULightmass::GatheredLightSample OutLightmapData[],
	int SizeX, int SizeY,
	int NumSamples)
{
	if(SizeX == 1022)
	SaveSampleDataToFile(SampleWorldPositions, SampleWorldNormals, TexelRadius, SizeX, SizeY, "samplecache.dat");

	float4* cudaSampleWorldPositions;
	float4* cudaSampleWorldNormals;
	float*	cudaTexelRadius;
	GPULightmass::GatheredLightSample* cudaOutLightmapData;

	cudaCheck(cudaMalloc((void**)&cudaSampleWorldPositions, SizeX * SizeY * sizeof(float4)));
	cudaCheck(cudaMemcpyAsync(cudaSampleWorldPositions, SampleWorldPositions, SizeX * SizeY * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaSampleWorldNormals, SizeX * SizeY * sizeof(float4)));
	cudaCheck(cudaMemcpyAsync(cudaSampleWorldNormals, SampleWorldNormals, SizeX * SizeY * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTexelRadius, SizeX * SizeY * sizeof(float)));
	cudaCheck(cudaMemcpyAsync(cudaTexelRadius, TexelRadius, SizeX * SizeY * sizeof(float), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaOutLightmapData, SizeX * SizeY * sizeof(GPULightmass::GatheredLightSample)));

	rtBindSampleData(cudaSampleWorldPositions, cudaSampleWorldNormals, cudaTexelRadius, cudaOutLightmapData, SizeX, SizeY);

	cudaCheck(cudaMemset(cudaOutLightmapData, 0, SizeX * SizeY * sizeof(GPULightmass::GatheredLightSample)));

	float MRaysPerSecond = 1.0f;
	float time = rtTimedLaunch(MRaysPerSecond, NumSamples);

	cudaCheck(cudaMemcpyAsync(OutLightmapData, cudaOutLightmapData, SizeX * SizeY * sizeof(GPULightmass::GatheredLightSample), cudaMemcpyDeviceToHost));

	cudaCheck(cudaFree(cudaOutLightmapData));
	cudaCheck(cudaFree(cudaSampleWorldPositions));
	cudaCheck(cudaFree(cudaSampleWorldNormals));
	cudaCheck(cudaFree(cudaTexelRadius));
}

void GenerateSDF(Vec3f BoundingBoxMin, Vec3f BoundingBoxMax, Vec3i VolumeDimension, float* OutBuffer)
{
	float* cudaOutBuffer;

	cudaCheck(cudaMalloc((void**)&cudaOutBuffer, VolumeDimension.x * VolumeDimension.y * 8 * sizeof(float)));

	int gridSizeZ = ceilf((float)VolumeDimension.z / 8);

	for (int ZSlice8 = 0; ZSlice8 < gridSizeZ; ZSlice8++)
	{
		cudaGenerateSignedDistanceFieldVolumeData(BoundingBoxMin, BoundingBoxMax, VolumeDimension, cudaOutBuffer, ZSlice8);
		int Size = 8 * (ZSlice8 + 1) >= VolumeDimension.z ? (VolumeDimension.z - 8 * (ZSlice8)) : 8;
		cudaCheck(cudaMemcpy(OutBuffer + VolumeDimension.x * VolumeDimension.y * 8 * ZSlice8, cudaOutBuffer, VolumeDimension.x * VolumeDimension.y * Size * sizeof(float), cudaMemcpyDeviceToHost));
	}
	
	cudaCheck(cudaFree(cudaOutBuffer));

}
