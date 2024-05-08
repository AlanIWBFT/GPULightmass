#pragma once

struct AABB
{
	float3 Low;
	float3 High;
};

__device__ AABB WorldPositionAABB;

__device__ __inline__ void atomicMin(F32* ptr, F32 value)
{
	U32 curr = atomicAdd((U32*)ptr, 0);
	while (value < __int_as_float(curr))
	{
		U32 prev = curr;
		curr = atomicCAS((U32*)ptr, curr, __float_as_int(value));
		if (curr == prev)
			break;
	}
}

__device__ __inline__ void atomicMax(F32* ptr, F32 value)
{
	U32 curr = atomicAdd((U32*)ptr, 0);
	while (value > __int_as_float(curr))
	{
		U32 prev = curr;
		curr = atomicCAS((U32*)ptr, curr, __float_as_int(value));
		if (curr == prev)
			break;
	}
}

__global__ void ComputeAABB()
{
	__shared__ AABB threadAABBs[64];

	threadAABBs[threadIdx.x].Low = make_float3(FLT_MAX);
	threadAABBs[threadIdx.x].High = make_float3(-FLT_MAX);

	const int TargetTexelLocation = blockIdx.x * blockDim.x + threadIdx.x;

	if (TargetTexelLocation >= BindedSizeX * BindedSizeY)
		return;

	float3 WorldPosition = make_float3(tex1Dfetch<float4>(SampleWorldPositionsTexture, TargetTexelLocation));
	float TexelRadius = tex1Dfetch<float>(TexelRadiusTexture, TargetTexelLocation);

	if (TexelRadius == 0.0f)
		return;

	threadAABBs[threadIdx.x].Low = WorldPosition;
	threadAABBs[threadIdx.x].High = WorldPosition;

	for (int stride = 64 >> 1; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
		{
			threadAABBs[threadIdx.x].Low = fminf(threadAABBs[threadIdx.x].Low, threadAABBs[threadIdx.x + stride].Low);
			threadAABBs[threadIdx.x].High = fmaxf(threadAABBs[threadIdx.x].High, threadAABBs[threadIdx.x + stride].High);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		atomicMin(&WorldPositionAABB.Low.x, threadAABBs[0].Low.x);
		atomicMin(&WorldPositionAABB.Low.y, threadAABBs[0].Low.y);
		atomicMin(&WorldPositionAABB.Low.z, threadAABBs[0].Low.z);
		atomicMax(&WorldPositionAABB.High.x, threadAABBs[0].High.x);
		atomicMax(&WorldPositionAABB.High.y, threadAABBs[0].High.y);
		atomicMax(&WorldPositionAABB.High.z, threadAABBs[0].High.z);
	}
}

struct MortonHash6
{
	int OriginalPosition;
	unsigned int Hash[6];
};

__device__ __inline__ void collectBits(MortonHash6 &OutHash, int index, float value)
{
	for (int i = 0; i < 32; i++)
		OutHash.Hash[(index + i * 6) >> 5] |= ((static_cast<unsigned int>(value) >> i) & 1) << ((index + i * 6) & 31);
}

__global__ void ComputeMortonHash(
	MortonHash6* OutHashes
)
{
	const int TargetTexelLocation = blockIdx.x * blockDim.x + threadIdx.x;

	if (TargetTexelLocation >= BindedSizeX * BindedSizeY)
		return;

	float3 WorldPosition = make_float3(tex1Dfetch<float4>(SampleWorldPositionsTexture, TargetTexelLocation));
	float3 WorldNormal = make_float3(tex1Dfetch<float4>(SampleWorldNormalsTexture, TargetTexelLocation));
	float TexelRadius = tex1Dfetch<float>(TexelRadiusTexture, TargetTexelLocation);

	if (TexelRadius == 0.0f)
		return;

	float3 a = (WorldPosition - WorldPositionAABB.Low) / fmaxf((WorldPositionAABB.High - WorldPositionAABB.Low), make_float3(0.00001f));
	float3 b = (normalize(WorldNormal) + 1.0f) * 0.5f;

	int OutSlot = atomicAggInc(&MappedTexelCounter);

	OutHashes[OutSlot].OriginalPosition = TargetTexelLocation;

	for (int i = 0; i < 6; i++)
		OutHashes[OutSlot].Hash[i] = 0;

	collectBits(OutHashes[OutSlot], 0, a.x * 256.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 1, a.y * 256.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 2, a.z * 256.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 3, b.x * 32.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 4, b.y * 32.0f * 65536.0f);
	collectBits(OutHashes[OutSlot], 5, b.z * 32.0f * 65536.0f);
}

__host__ int compareMortonKey(const void* A, const void* B)
{
	const MortonHash6& a = *((const MortonHash6*)A);
	const MortonHash6& b = *((const MortonHash6*)B);
	if (a.Hash[5] != b.Hash[5]) return (a.Hash[5] < b.Hash[5] ? 1 : -1);
	if (a.Hash[4] != b.Hash[4]) return (a.Hash[4] < b.Hash[4] ? 1 : -1);
	if (a.Hash[3] != b.Hash[3]) return (a.Hash[3] < b.Hash[3] ? 1 : -1);
	if (a.Hash[2] != b.Hash[2]) return (a.Hash[2] < b.Hash[2] ? 1 : -1);
	if (a.Hash[1] != b.Hash[1]) return (a.Hash[1] < b.Hash[1] ? 1 : -1);
	if (a.Hash[0] != b.Hash[0]) return (a.Hash[0] < b.Hash[0] ? 1 : -1);
	return 0;
}
