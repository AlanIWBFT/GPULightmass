/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

__global__ void rtTraceDynamicFetch()
{
	const bool IsShadowRay = false;
	const int EntrypointSentinel = 0x76543210;
	const int STACK_SIZE = 32;

	int traversalStack[STACK_SIZE];

	int     rayidx;
	float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
	float   idirx, idiry, idirz;    // 1 / ray direction
	float   oodx, oody, oodz;       // ray origin / ray direction
	float2  triangleuv;
	float3	trianglenormal;

	int*    stackPtr;               // Current position in traversal stack.
	int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
	int     nodeAddr = EntrypointSentinel;
	int     hitAddr;               // Triangle index of the closest intersection, -1 if none.
	float   hitT;                   // t-value of the closest intersection.

	__shared__ volatile int nextRayArray[2];

	do
	{
		const int tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];

		// Fetch new rays from the global pool using lane 0.

		const bool          terminated = nodeAddr == EntrypointSentinel;
		const unsigned int  maskTerminated = __ballot_sync(__activemask(), terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		if (terminated)
		{
			if (idxTerminated == 0)
				rayBase = atomicAdd(&FinishedRayCount, numTerminated);

			rayidx = rayBase + idxTerminated;
			if (rayidx >= RayCount)
				break;

			float4 RayOrigin = RayBuffer[rayidx].RayOriginAndNearClip;
			float4 RayDirection = RayBuffer[rayidx].RayDirectionAndFarClip;
			origx = RayOrigin.x;
			origy = RayOrigin.y;
			origz = RayOrigin.z;
			dirx = RayDirection.x;
			diry = RayDirection.y;
			dirz = RayDirection.z;
			tmin = RayOrigin.w;

			// ooeps is very small number, used instead of raydir xyz component when that component is near zero
			float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
			idirx = 1.0f / (fabsf(RayDirection.x) > ooeps ? RayDirection.x : copysignf(ooeps, RayDirection.x)); // inverse ray direction
			idiry = 1.0f / (fabsf(RayDirection.y) > ooeps ? RayDirection.y : copysignf(ooeps, RayDirection.y)); // inverse ray direction
			idirz = 1.0f / (fabsf(RayDirection.z) > ooeps ? RayDirection.z : copysignf(ooeps, RayDirection.z)); // inverse ray direction
			oodx = origx * idirx;  // ray origin / ray direction
			oody = origy * idiry;  // ray origin / ray direction
			oodz = origz * idirz;  // ray origin / ray direction

								   // Setup traversal + initialisation

			traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
			stackPtr = &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
			leafAddr = 0;   // No postponed leaf.
			nodeAddr = 0;   // Start from the root.
			hitAddr = -1;  // No triangle intersected so far.
			hitT = RayDirection.w; // tmax  
		}

		// Traversal loop.

		while (nodeAddr != EntrypointSentinel)
		{
			// Traverse internal nodes until all SIMD lanes have found a leaf.

			while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)   // functionally equivalent, but faster
			{
				// Fetch AABBs of the two child nodes.

				const float4 n0xy = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 0); // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				const float4 n1xy = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 1); // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				const float4 nz = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 2); // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				float4 tmp = tex1Dfetch(BVHTreeNodesTexture, nodeAddr + 3); // child_index0, child_index1
				int2  cnodes = *(int2*)&tmp;

				// Intersect the ray against the child nodes.

				const float c0lox = n0xy.x * idirx - oodx;
				const float c0hix = n0xy.y * idirx - oodx;
				const float c0loy = n0xy.z * idiry - oody;
				const float c0hiy = n0xy.w * idiry - oody;
				const float c0loz = nz.x   * idirz - oodz;
				const float c0hiz = nz.y   * idirz - oodz;
				const float c1loz = nz.z   * idirz - oodz;
				const float c1hiz = nz.w   * idirz - oodz;
				const float c0min = tMinFermi(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
				const float c0max = tMaxFermi(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
				const float c1lox = n1xy.x * idirx - oodx;
				const float c1hix = n1xy.y * idirx - oodx;
				const float c1loy = n1xy.z * idiry - oody;
				const float c1hiy = n1xy.w * idiry - oody;
				const float c1min = tMinFermi(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
				const float c1max = tMaxFermi(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

				bool swp = (c1min < c0min);

				bool traverseChild0 = (c0max >= c0min);
				bool traverseChild1 = (c1max >= c1min);

				// Neither child was intersected => pop stack.

				if (!traverseChild0 && !traverseChild1)
				{
					nodeAddr = *stackPtr;
					stackPtr--;
				}

				// Otherwise => fetch child pointers.

				else
				{
					nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

					// Both children were intersected => push the farther one.

					if (traverseChild0 && traverseChild1)
					{
						if (swp)
							swap(nodeAddr, cnodes.y);
						stackPtr++;
						*stackPtr = cnodes.y;
					}
				}

				// First leaf => postpone and continue traversal.

				if (nodeAddr < 0 && leafAddr >= 0)     // Postpone max 1
													   //              if (nodeAddr < 0 && leafAddr2 >= 0)     // Postpone max 2
				{
					//leafAddr2= leafAddr;          // postpone 2
					leafAddr = nodeAddr;
					nodeAddr = *stackPtr;
					stackPtr--;
				}

				// All SIMD lanes have found a leaf? => process them.

				if (!__any_sync(__activemask(), leafAddr >= 0))
					break;
			}

			// Process postponed leaf nodes.

			while (leafAddr < 0)
			{
				for (int triAddr = ~leafAddr;; triAddr += 3)
				{
					float4 v00 = tex1Dfetch(TriangleWoopCoordinatesTexture, triAddr);
					float4 v11 = tex1Dfetch(TriangleWoopCoordinatesTexture, triAddr + 1);
					float4 v22 = tex1Dfetch(TriangleWoopCoordinatesTexture, triAddr + 2);

					if (__float_as_int(v00.x) == 0x80000000)
						break;

					float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
					float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
					float t = Oz * invDz;

					if (t > tmin && t < hitT)
					{
						float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
						float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;
						float u = Ox + t * Dx;

						if (u >= 0.0f && u <= 1.0f)
						{
							float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
							float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
							float v = Oy + t*Dy;

							if (v >= 0.0f && u + v <= 1.0f)
							{
								int triangleIndex = tex1Dfetch(MappingFromTriangleAddressToIndexTexture, triAddr);
								int materialIndex = tex1Dfetch(MappingFromTriangleAddressToIndexTexture, triAddr + 1);
								bool isHitRejected = false;

								if (materialIndex >= 0)
								{
									int I0 = TriangleIndexBuffer[triangleIndex * 3 + 0];
									int I1 = TriangleIndexBuffer[triangleIndex * 3 + 1];
									int I2 = TriangleIndexBuffer[triangleIndex * 3 + 2];
									float2 UV0 = VertexTextureUVs[I0];
									float2 UV1 = VertexTextureUVs[I1];
									float2 UV2 = VertexTextureUVs[I2];
									float2 FinalUV = UV0 * u + UV1 * v + UV2 * (1.0f - u - v);

									float Mask = tex2D<float>(MaskedCollisionMaps[materialIndex], FinalUV.x, FinalUV.y);

									if (Mask < 0.5f)
										isHitRejected = true;
								}

								if (!isHitRejected)
								{
									trianglenormal = cross(make_float3(v22.x, v22.y, v22.z), make_float3(v11.x, v11.y, v11.z));
									triangleuv.x = u;
									triangleuv.y = v;

									hitT = t;
									hitAddr = triAddr;

									if (IsShadowRay || FORCE_SHADOWRAYS)
									{
										nodeAddr = EntrypointSentinel;
										break;
									}
								}
							}
						}
					}
				}

				leafAddr = nodeAddr;
				if (nodeAddr < 0)
				{
					nodeAddr = *stackPtr;
					stackPtr--;
				}
			}

			// Force reconvergence point here
			if (__popc(__activemask()) < 0)
				break;
		}

		if (terminated)
		{
			RayHitResultBuffer[rayidx].TriangleUV = triangleuv;
			RayHitResultBuffer[rayidx].HitTriangleIndex = hitAddr != -1 ? tex1Dfetch(MappingFromTriangleAddressToIndexTexture, hitAddr) : -1;
			RayHitResultBuffer[rayidx].TriangleNormalUnnormalized = trianglenormal;
		}

	} while (true);
}
