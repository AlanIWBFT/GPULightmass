#pragma once

#include <embree4/rtcore.h>
#include <vector>

template <int N>
class BVHNNode
{
public:
	float3 ChildrenMin[N];
	float3 ChildrenMax[N];
	BVHNNode* Children[N] = {};
	int ChildCount = 0;
	int* LeafPrimitiveRefs = nullptr;
	int LeafPrimitiveCount = 0;
};

typedef BVHNNode<2> BVH2Node;
typedef BVHNNode<8> BVH8Node;

struct MeshDescription
{
	const int3* IndexBuffer;
	const float3* VertexBuffer;
};

class EmbreeBVHBuilder
{
public:
	EmbreeBVHBuilder(
		const int InNumVertices,
		const int InNumTriangles,
		const float3 InVertexWorldPositionBuffer[],
		const int3 InTriangleIndexBuffer[]);

	~EmbreeBVHBuilder();

	BVH2Node* BuildBVH2();
	BVH8Node* BuildBVH8();

	void ConvertToCUDABVH2(
		BVH2Node * root,
		const int TriangleMaterialIndex[],
		std::vector<float4>& OutNodeData,
		std::vector<float4>& OutWoopifiedTriangles,
		std::vector<int>& OutTriangleIndices);
	void ConvertToCUDABVH8(
		BVH8Node* root,
		const int TriangleMaterialIndex[],
		std::vector<float4>& OutNodeData,
		std::vector<float4>& OutWoopifiedTriangles,
		std::vector<int>& OutTriangleIndices);

private:
	RTCDevice EmbreeDevice;

	const int NumVertices;
	const int NumTriangles;
	const float3* VertexWorldPositionBuffer;
	const int3* TriangleIndexBuffer;
};
