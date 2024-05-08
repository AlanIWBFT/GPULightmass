#define EMBREE_STATIC_LIB

#include "linear_math.h"
#include "helper_math.h"
#include "../rt/rtDebugFunc.h"
#include "EmbreeBVHBuilder.h"
#include "../ProgressReport.h"
#include <tbb/parallel_for.h>
#undef NDEBUG
#include <cassert>

#pragma comment(lib, "embree_avx.lib")
#pragma comment(lib, "embree_avx2.lib")
#pragma comment(lib, "embree_sse42.lib")
#pragma comment(lib, "embree4.lib")
#pragma comment(lib, "lexers.lib")
#pragma comment(lib, "math.lib")
#pragma comment(lib, "simd.lib")
#pragma comment(lib, "sys.lib")
#pragma comment(lib, "tasking.lib")

/* Callback to create a node */
template <int N>
void* CreateNode(RTCThreadLocalAllocator allocator, unsigned int childCount, void* userPtr)
{
	BVHNNode<N>* newNode = new (rtcThreadLocalAlloc(allocator, sizeof(BVHNNode<N>), 16)) BVHNNode<N>;
	return newNode;
}

/* Callback to set the pointer to all children */
template <int N>
void SetNodeChildren(void* nodePtr, void** children, unsigned int childCount, void* userPtr)
{
	assert(childCount > 0);
	BVHNNode<N>* node = (BVHNNode<N>*)nodePtr;
	for (int i = 0; i < childCount; i++)
		node->Children[i] = ((BVHNNode<N>**)children)[i];
	node->ChildCount = childCount;
}

/* Callback to set the bounds of all children */
template <int N>
void SetNodeBounds(void* nodePtr, const struct RTCBounds** bounds, unsigned int childCount, void* userPtr)
{
	BVHNNode<N>* node = (BVHNNode<N>*)nodePtr;
	for (int i = 0; i < childCount; i++)
	{
		node->ChildrenMin[i].x = bounds[i]->lower_x;
		node->ChildrenMin[i].y = bounds[i]->lower_y;
		node->ChildrenMin[i].z = bounds[i]->lower_z;
		node->ChildrenMax[i].x = bounds[i]->upper_x;
		node->ChildrenMax[i].y = bounds[i]->upper_y;
		node->ChildrenMax[i].z = bounds[i]->upper_z;
	}
}

/* Callback to create a leaf node */
template <int N>
void* CreateLeaf(RTCThreadLocalAllocator allocator, const struct RTCBuildPrimitive* primitives, size_t primitiveCount, void* userPtr)
{
	assert(primitiveCount > 0);
	BVHNNode<N>* newNode = new (rtcThreadLocalAlloc(allocator, sizeof(BVHNNode<N>), 16)) BVHNNode<N>;
	newNode->LeafPrimitiveRefs = new (rtcThreadLocalAlloc(allocator, sizeof(int) * primitiveCount, sizeof(int))) int[primitiveCount]();
	for (int i = 0; i < primitiveCount; i++)
		newNode->LeafPrimitiveRefs[i] = primitives[i].primID;
	newNode->LeafPrimitiveCount = primitiveCount;
	return newNode;
}

/* Callback to split a build primitive */
void SplitPrimitive(const struct RTCBuildPrimitive* primitive, unsigned int dimension, float position, struct RTCBounds* leftBounds, struct RTCBounds* rightBounds, void* userPtr)
{
	leftBounds->lower_x = primitive->lower_x;
	leftBounds->lower_y = primitive->lower_y;
	leftBounds->lower_z = primitive->lower_z;
	leftBounds->upper_x = primitive->upper_x;
	leftBounds->upper_y = primitive->upper_y;
	leftBounds->upper_z = primitive->upper_z;

	rightBounds->lower_x = primitive->lower_x;
	rightBounds->lower_y = primitive->lower_y;
	rightBounds->lower_z = primitive->lower_z;
	rightBounds->upper_x = primitive->upper_x;
	rightBounds->upper_y = primitive->upper_y;
	rightBounds->upper_z = primitive->upper_z;

	switch (dimension)
	{
	case 0:
		leftBounds->upper_x = rightBounds->lower_x = position;
		break;
	case 1:
		leftBounds->upper_y = rightBounds->lower_y = position;
		break;
	case 2:
		leftBounds->upper_z = rightBounds->lower_z = position;
		break;
	default:
		assert(0);
	}
}

float GetDimension(float3 v, unsigned int dimension)
{
	switch (dimension)
	{
	case 0:
		return v.x;
	case 1:
		return v.y;
	case 2:
		return v.z;
	default:
		assert(0);
		return 0;
	}
}

struct AABB
{
	float3 Low;
	float3 High;
};

void SplitTriangle(const int3* IndexBuffer, const float3* VertexBuffer, int TriangleIndex, unsigned int Dimension, float Position, AABB& InOutLeft, AABB& InOutRight)
{
	AABB NewLeft;
	AABB NewRight;
	NewLeft.Low = make_float3(FLT_MAX);
	NewLeft.High = make_float3(-FLT_MAX);
	NewRight.Low = make_float3(FLT_MAX);
	NewRight.High = make_float3(-FLT_MAX);

	{
		float3 V0 = VertexBuffer[IndexBuffer[TriangleIndex].z];
		float3 V1 = VertexBuffer[IndexBuffer[TriangleIndex].x];

		float V0p = GetDimension(V0, Dimension);
		float V1p = GetDimension(V1, Dimension);

		if (V0p <= Position)
		{
			NewLeft.Low = fminf(NewLeft.Low, V0);
			NewLeft.High = fmaxf(NewLeft.High, V0);
		}

		if (V0p >= Position)
		{
			NewRight.Low = fminf(NewRight.Low, V0);
			NewRight.High = fmaxf(NewRight.High, V0);
		}

		if ((V0p < Position && V1p > Position) || (V0p > Position && V1p < Position))
		{
			float3 t = lerp(V0, V1, clamp((Position - V0p) / (V1p - V0p), 0.0f, 1.0f));
			NewLeft.Low = fminf(NewLeft.Low, t);
			NewLeft.High = fmaxf(NewLeft.High, t);
			NewRight.Low = fminf(NewRight.Low, t);
			NewRight.High = fmaxf(NewRight.High, t);
		}
	}

	{
		float3 V0 = VertexBuffer[IndexBuffer[TriangleIndex].x];
		float3 V1 = VertexBuffer[IndexBuffer[TriangleIndex].y];

		float V0p = GetDimension(V0, Dimension);
		float V1p = GetDimension(V1, Dimension);

		if (V0p <= Position)
		{
			NewLeft.Low = fminf(NewLeft.Low, V0);
			NewLeft.High = fmaxf(NewLeft.High, V0);
		}

		if (V0p >= Position)
		{
			NewRight.Low = fminf(NewRight.Low, V0);
			NewRight.High = fmaxf(NewRight.High, V0);
		}

		if ((V0p < Position && V1p > Position) || (V0p > Position && V1p < Position))
		{
			float3 t = lerp(V0, V1, clamp((Position - V0p) / (V1p - V0p), 0.0f, 1.0f));
			NewLeft.Low = fminf(NewLeft.Low, t);
			NewLeft.High = fmaxf(NewLeft.High, t);
			NewRight.Low = fminf(NewRight.Low, t);
			NewRight.High = fmaxf(NewRight.High, t);
		}
	}

	{
		float3 V0 = VertexBuffer[IndexBuffer[TriangleIndex].y];
		float3 V1 = VertexBuffer[IndexBuffer[TriangleIndex].z];

		float V0p = GetDimension(V0, Dimension);
		float V1p = GetDimension(V1, Dimension);

		if (V0p <= Position)
		{
			NewLeft.Low = fminf(NewLeft.Low, V0);
			NewLeft.High = fmaxf(NewLeft.High, V0);
		}

		if (V0p >= Position)
		{
			NewRight.Low = fminf(NewRight.Low, V0);
			NewRight.High = fmaxf(NewRight.High, V0);
		}

		if ((V0p < Position && V1p > Position) || (V0p > Position && V1p < Position))
		{
			float3 t = lerp(V0, V1, clamp((Position - V0p) / (V1p - V0p), 0.0f, 1.0f));
			NewLeft.Low = fminf(NewLeft.Low, t);
			NewLeft.High = fmaxf(NewLeft.High, t);
			NewRight.Low = fminf(NewRight.Low, t);
			NewRight.High = fmaxf(NewRight.High, t);
		}
	}

	InOutLeft.Low = fmaxf(InOutLeft.Low, NewLeft.Low);
	InOutLeft.High = fminf(InOutLeft.High, NewLeft.High);
	InOutRight.Low = fmaxf(InOutRight.Low, NewRight.Low);
	InOutRight.High = fminf(InOutRight.High, NewRight.High);
}

void SplitPrimitiveTriangle(const struct RTCBuildPrimitive* primitive, unsigned int dimension, float position, struct RTCBounds* leftBounds, struct RTCBounds* rightBounds, void* userPtr)
{
	AABB Left;
	Left.Low.x = primitive->lower_x;
	Left.Low.y = primitive->lower_y;
	Left.Low.z = primitive->lower_z;
	Left.High.x = primitive->upper_x;
	Left.High.y = primitive->upper_y;
	Left.High.z = primitive->upper_z;

	AABB Right;
	Right.Low.x  = primitive->lower_x;
	Right.Low.y  = primitive->lower_y;
	Right.Low.z  = primitive->lower_z;
	Right.High.x = primitive->upper_x;
	Right.High.y = primitive->upper_y;
	Right.High.z = primitive->upper_z;

	MeshDescription* Mesh = (MeshDescription*)userPtr;

	SplitTriangle(Mesh->IndexBuffer, Mesh->VertexBuffer, primitive->primID, dimension, position, Left, Right);

	leftBounds->lower_x = Left.Low.x;
	leftBounds->lower_y = Left.Low.y;
	leftBounds->lower_z = Left.Low.z;
	leftBounds->upper_x = Left.High.x;
	leftBounds->upper_y = Left.High.y;
	leftBounds->upper_z = Left.High.z;

	rightBounds->lower_x = Right.Low.x;
	rightBounds->lower_y = Right.Low.y;
	rightBounds->lower_z = Right.Low.z;
	rightBounds->upper_x = Right.High.x;
	rightBounds->upper_y = Right.High.y;
	rightBounds->upper_z = Right.High.z;
}

bool ProgressMonitor(void* ptr, double n)
{
	char progressTextBuffer[256];
	char duplicateTextBuffer[256];
	sprintf(progressTextBuffer, "Building BVH %.0lf%%", n * 100);
	sprintf(duplicateTextBuffer, "");

	ReportProgress(
		progressTextBuffer,
		n * 100,
		duplicateTextBuffer,
		0,
		false
	);
	return true;
}

EmbreeBVHBuilder::EmbreeBVHBuilder(
	const int InNumVertices,
	const int InNumTriangles,
	const float3 InVertexWorldPositionBuffer[],
	const int3 InTriangleIndexBuffer[])
	:
	NumVertices(InNumVertices),
	NumTriangles(InNumTriangles),
	VertexWorldPositionBuffer(InVertexWorldPositionBuffer),
	TriangleIndexBuffer(InTriangleIndexBuffer)
{
	EmbreeDevice = rtcNewDevice(nullptr);
}

EmbreeBVHBuilder::~EmbreeBVHBuilder()
{
	rtcReleaseDevice(EmbreeDevice);
}

BVH2Node * EmbreeBVHBuilder::BuildBVH2()
{
	RTCBuildPrimitive* Primitives = (RTCBuildPrimitive*)_aligned_malloc(sizeof(RTCBuildPrimitive) * NumTriangles * 2, 32);

	tbb::parallel_for(0, NumTriangles, [&](int PrimitiveIndex)
	{
		Primitives[PrimitiveIndex].lower_x = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].x, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].x), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].x);
		Primitives[PrimitiveIndex].lower_y = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].y, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].y), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].y);
		Primitives[PrimitiveIndex].lower_z = fminf(fminf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].z, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].z), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].z);
		Primitives[PrimitiveIndex].upper_x = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].x, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].x), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].x);
		Primitives[PrimitiveIndex].upper_y = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].y, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].y), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].y);
		Primitives[PrimitiveIndex].upper_z = fmaxf(fmaxf(VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].x].z, VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].y].z), VertexWorldPositionBuffer[TriangleIndexBuffer[PrimitiveIndex].z].z);
		Primitives[PrimitiveIndex].geomID = 0;
		Primitives[PrimitiveIndex].primID = PrimitiveIndex;
	});

	RTCBVH bvh = rtcNewBVH(EmbreeDevice);
	RTCBuildArguments args = rtcDefaultBuildArguments();
	args.buildQuality = RTC_BUILD_QUALITY_HIGH;
	args.maxDepth = 40;
	args.bvh = bvh;
	args.primitives = Primitives;
	args.primitiveCount = NumTriangles;
	args.primitiveArrayCapacity = NumTriangles * 2;
	args.createNode = CreateNode<2>;
	args.setNodeChildren = SetNodeChildren<2>;
	args.setNodeBounds = SetNodeBounds<2>;
	args.createLeaf = CreateLeaf<2>;

	MeshDescription Mesh { TriangleIndexBuffer, VertexWorldPositionBuffer };
	args.userPtr = &Mesh;
	args.splitPrimitive = SplitPrimitiveTriangle;

	args.buildProgress = ProgressMonitor;

	BVH2Node* root = (BVH2Node*)rtcBuildBVH(&args);

	ReportProgress(
		"Building BVH 100%",
		100,
		"",
		0,
		true
	);

	_aligned_free(Primitives);
	return root;
}

void EmbreeBVHBuilder::ConvertToCUDABVH2(
	const BVH2Node * root,
	const int TriangleMaterialIndex[],
	std::vector<float4>& OutNodeData,
	std::vector<float4>& OutWoopifiedTriangles,
	std::vector<int>& OutTriangleIndices
)
{
	const BVH2Node* stackNodePtr[32];
	int stackNodeParentIdx[32];
	int stackNodeIdxInParent[32];
	int stackPtr = 0;

	stackNodePtr[stackPtr] = root;
	stackNodeParentIdx[stackPtr] = -1;
	stackNodeIdxInParent[stackPtr] = 0;
	stackPtr++;

	while (stackPtr > 0)
	{
		stackPtr--;
		const BVH2Node* node = stackNodePtr[stackPtr];
		const int parentNodeAddr = stackNodeParentIdx[stackPtr];
		const int nodeIdxInParent = stackNodeIdxInParent[stackPtr];

		if (node->LeafPrimitiveCount == 0)
		{
			assert(node->Children[0] != nullptr && node->Children[1] != nullptr);

			int currentNodeAddr = OutNodeData.size();
			if (parentNodeAddr >= 0)
			{
				if (stackNodeIdxInParent[stackPtr] == 0)
					OutNodeData[parentNodeAddr + 3].x = *(float*)&currentNodeAddr;
				else if (stackNodeIdxInParent[stackPtr] == 1)
					OutNodeData[parentNodeAddr + 3].y = *(float*)&currentNodeAddr;
				else
					assert(0);
			}

			OutNodeData.push_back(make_float4(node->ChildrenMin[0].x, node->ChildrenMax[0].x, node->ChildrenMin[0].y, node->ChildrenMax[0].y));
			OutNodeData.push_back(make_float4(node->ChildrenMin[1].x, node->ChildrenMax[1].x, node->ChildrenMin[1].y, node->ChildrenMax[1].y));
			OutNodeData.push_back(make_float4(node->ChildrenMin[0].z, node->ChildrenMax[0].z, node->ChildrenMin[1].z, node->ChildrenMax[1].z));
			OutNodeData.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.0f)); // children indices, will be filled in by the children themselves

			stackNodePtr[stackPtr] = node->Children[0];
			stackNodeParentIdx[stackPtr] = currentNodeAddr;
			stackNodeIdxInParent[stackPtr] = 0;
			stackPtr++;

			stackNodePtr[stackPtr] = node->Children[1];
			stackNodeParentIdx[stackPtr] = currentNodeAddr;
			stackNodeIdxInParent[stackPtr] = 1;
			stackPtr++;
		}
		else
		{
			assert(node->Children[0] == nullptr && node->Children[1] == nullptr);
			// Leaf node
			int currentNodeAddr = OutWoopifiedTriangles.size();
			currentNodeAddr = ~currentNodeAddr;

			if (parentNodeAddr >= 0)
			{
				if (stackNodeIdxInParent[stackPtr] == 0)
					OutNodeData[parentNodeAddr + 3].x = *(float*)&currentNodeAddr;
				else if (stackNodeIdxInParent[stackPtr] == 1)
					OutNodeData[parentNodeAddr + 3].y = *(float*)&currentNodeAddr;
				else
					assert(0);
			}

			for (int i = 0; i < node->LeafPrimitiveCount; i++)
			{
				int triangleIndex = node->LeafPrimitiveRefs[i];
				float3 v0 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].x];
				float3 v1 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].y];
				float3 v2 = VertexWorldPositionBuffer[TriangleIndexBuffer[triangleIndex].z];

				Mat4f mtx;
				float4 col0 = make_float4(v0 - v2, 0.0f);
				float4 col1 = make_float4(v1 - v2, 0.0f);
				float4 col2 = make_float4(cross(v0 - v2, v1 - v2), 0.0f);
				float4 col3 = make_float4(v2, 1.0f);
				mtx.setCol(0, Vec4f(col0)); // sets matrix column 0 equal to a Vec4f(Vec3f, 0.0f )
				mtx.setCol(1, Vec4f(col1));
				mtx.setCol(2, Vec4f(col2));
				mtx.setCol(3, Vec4f(col3));
				mtx = invert(mtx);

				float4 WoopifiedVertices[3];
				WoopifiedVertices[0] = make_float4(mtx(2, 0), mtx(2, 1), mtx(2, 2), -mtx(2, 3));
				WoopifiedVertices[1] = make_float4(mtx.getRow(0).x, mtx.getRow(0).y, mtx.getRow(0).z, mtx.getRow(0).w);
				WoopifiedVertices[2] = make_float4(mtx.getRow(1).x, mtx.getRow(1).y, mtx.getRow(1).z, mtx.getRow(1).w);

				OutWoopifiedTriangles.push_back(WoopifiedVertices[0]);
				OutWoopifiedTriangles.push_back(WoopifiedVertices[1]);
				OutWoopifiedTriangles.push_back(WoopifiedVertices[2]);

				OutTriangleIndices.push_back(triangleIndex);
				OutTriangleIndices.push_back(TriangleMaterialIndex[triangleIndex]);
				OutTriangleIndices.push_back(0);
			}

			unsigned int leafTerminator = 0x80000000;
			OutWoopifiedTriangles.push_back(make_float4(*(float*)(&leafTerminator), 0, 0, 0));

			OutTriangleIndices.push_back(0);
		}
	}
}
