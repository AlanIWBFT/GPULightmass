#pragma once

__device__ void rtTrace(
	HitInfo& OutHitInfo,
	const float4 RayOrigin,
	const float4 RayDirection,
	bool IsShadowRay)
{
	const float ooeps = exp2f(-80.0f);

	const int STACK_SIZE = 32;
	uint2 traversalStack[STACK_SIZE];

	float3 orig = make_float3(RayOrigin);
	float3 dir = make_float3(RayDirection);
	float tmin = RayOrigin.w;
	float tmax = RayDirection.w;
	float idirx = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x)); // inverse ray direction
	float idiry = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y)); // inverse ray direction
	float idirz = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z)); // inverse ray direction
	uint octinv = ((dir.x < 0 ? 1 : 0) << 2) | ((dir.y < 0 ? 1 : 0) << 1) | ((dir.z < 0 ? 1 : 0) << 0);
	octinv = 7 - octinv;
	uint2 nodeGroup = make_uint2(0, 0b10000000000000000000000000000000);
	uint2 triangleGroup = make_uint2(0);
	char stackPtr = 0;
	int hitAddr = -1;
	float2 triangleuv;
	float3 trianglenormal;

	do
	{
		if (nodeGroup.y > 0x00FFFFFF)
		{
			const unsigned int hits = nodeGroup.y;
			const unsigned int imask = nodeGroup.y;
			const unsigned int child_bit_index = __bfind(hits);
			const unsigned int child_node_base_index = nodeGroup.x;

			nodeGroup.y &= ~(1 << child_bit_index);

			if (nodeGroup.y > 0x00FFFFFF)
			{
				traversalStack[stackPtr++] = nodeGroup;
			}

			{
				const unsigned int slot_index = (child_bit_index - 24) ^ octinv;
				const unsigned int octinv4 = octinv * 0x01010101u;
				const unsigned int relative_index = __popc(imask & ~(0xFFFFFFFF << slot_index));
				const unsigned int child_node_index = child_node_base_index + relative_index;

				float4 n0, n1, n2, n3, n4;

				n0 = __ldg(BVHTreeNodes + child_node_index * 5 + 0);
				n1 = __ldg(BVHTreeNodes + child_node_index * 5 + 1);
				n2 = __ldg(BVHTreeNodes + child_node_index * 5 + 2);
				n3 = __ldg(BVHTreeNodes + child_node_index * 5 + 3);
				n4 = __ldg(BVHTreeNodes + child_node_index * 5 + 4);

				float3 p = make_float3(n0);
				int3 e;
				e.x = *((char*)&n0.w + 0);
				e.y = *((char*)&n0.w + 1);
				e.z = *((char*)&n0.w + 2);

				nodeGroup.x = __float_as_uint(n1.x);
				triangleGroup.x = __float_as_uint(n1.y);
				triangleGroup.y = 0;
				unsigned int hitmask = 0;

				const float adjusted_idirx = __uint_as_float((e.x + 127) << 23) * idirx;
				const float adjusted_idiry = __uint_as_float((e.y + 127) << 23) * idiry;
				const float adjusted_idirz = __uint_as_float((e.z + 127) << 23) * idirz;
				const float origx = -(orig.x - p.x) * idirx;
				const float origy = -(orig.y - p.y) * idiry;
				const float origz = -(orig.z - p.z) * idirz;

				{
					// First 4
					const unsigned int meta4 = __float_as_uint(n1.z);
					const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
					const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
					const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
					const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

					// Potential micro-optimization: use PRMT to do the selection here, as described by the paper
					uint swizzledLox = (idirx < 0) ? __float_as_uint(n3.z) : __float_as_uint(n2.x);
					uint swizzledHix = (idirx < 0) ? __float_as_uint(n2.x) : __float_as_uint(n3.z);

					uint swizzledLoy = (idiry < 0) ? __float_as_uint(n4.x) : __float_as_uint(n2.z);
					uint swizzledHiy = (idiry < 0) ? __float_as_uint(n2.z) : __float_as_uint(n4.x);

					uint swizzledLoz = (idirz < 0) ? __float_as_uint(n4.z) : __float_as_uint(n3.x);
					uint swizzledHiz = (idirz < 0) ? __float_as_uint(n3.x) : __float_as_uint(n4.z);

					float tminx[4];
					float tminy[4];
					float tminz[4];
					float tmaxx[4];
					float tmaxy[4];
					float tmaxz[4];

					tminx[0] = ((swizzledLox >> 0) & 0xFF) * adjusted_idirx + origx;
					tminx[1] = ((swizzledLox >> 8) & 0xFF) * adjusted_idirx + origx;
					tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
					tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

					tminy[0] = ((swizzledLoy >> 0) & 0xFF) * adjusted_idiry + origy;
					tminy[1] = ((swizzledLoy >> 8) & 0xFF) * adjusted_idiry + origy;
					tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
					tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

					tminz[0] = ((swizzledLoz >> 0) & 0xFF) * adjusted_idirz + origz;
					tminz[1] = ((swizzledLoz >> 8) & 0xFF) * adjusted_idirz + origz;
					tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
					tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

					tmaxx[0] = ((swizzledHix >> 0) & 0xFF) * adjusted_idirx + origx;
					tmaxx[1] = ((swizzledHix >> 8) & 0xFF) * adjusted_idirx + origx;
					tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
					tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

					tmaxy[0] = ((swizzledHiy >> 0) & 0xFF) * adjusted_idiry + origy;
					tmaxy[1] = ((swizzledHiy >> 8) & 0xFF) * adjusted_idiry + origy;
					tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
					tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

					tmaxz[0] = ((swizzledHiz >> 0) & 0xFF) * adjusted_idirz + origz;
					tmaxz[1] = ((swizzledHiz >> 8) & 0xFF) * adjusted_idirz + origz;
					tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
					tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

					for (int childIndex = 0; childIndex < 4; childIndex++)
					{
						// Use VMIN, VMAX to compute the slabs
						const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
						const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

						bool intersected = cmin <= cmax;

						// Potential micro-optimization: use VSHL to implement this part, as described by the paper
						if (intersected)
						{
							const unsigned int child_bits = extract_byte(child_bits4, childIndex);
							const unsigned int bit_index = extract_byte(bit_index4, childIndex);
							hitmask |= child_bits << bit_index;
						}
					}
				}

				{
					// Second 4
					const unsigned int meta4 = __float_as_uint(n1.w);
					const unsigned int is_inner4 = (meta4 & (meta4 << 1)) & 0x10101010;
					const unsigned int inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
					const unsigned int bit_index4 = (meta4 ^ (octinv4 & inner_mask4)) & 0x1F1F1F1F;
					const unsigned int child_bits4 = (meta4 >> 5) & 0x07070707;

					// Potential micro-optimization: use PRMT to do the selection here, as described by the paper
					uint swizzledLox = (idirx < 0) ? __float_as_uint(n3.w) : __float_as_uint(n2.y);
					uint swizzledHix = (idirx < 0) ? __float_as_uint(n2.y) : __float_as_uint(n3.w);

					uint swizzledLoy = (idiry < 0) ? __float_as_uint(n4.y) : __float_as_uint(n2.w);
					uint swizzledHiy = (idiry < 0) ? __float_as_uint(n2.w) : __float_as_uint(n4.y);

					uint swizzledLoz = (idirz < 0) ? __float_as_uint(n4.w) : __float_as_uint(n3.y);
					uint swizzledHiz = (idirz < 0) ? __float_as_uint(n3.y) : __float_as_uint(n4.w);

					float tminx[4];
					float tminy[4];
					float tminz[4];
					float tmaxx[4];
					float tmaxy[4];
					float tmaxz[4];

					tminx[0] = ((swizzledLox >> 0) & 0xFF) * adjusted_idirx + origx;
					tminx[1] = ((swizzledLox >> 8) & 0xFF) * adjusted_idirx + origx;
					tminx[2] = ((swizzledLox >> 16) & 0xFF) * adjusted_idirx + origx;
					tminx[3] = ((swizzledLox >> 24) & 0xFF) * adjusted_idirx + origx;

					tminy[0] = ((swizzledLoy >> 0) & 0xFF) * adjusted_idiry + origy;
					tminy[1] = ((swizzledLoy >> 8) & 0xFF) * adjusted_idiry + origy;
					tminy[2] = ((swizzledLoy >> 16) & 0xFF) * adjusted_idiry + origy;
					tminy[3] = ((swizzledLoy >> 24) & 0xFF) * adjusted_idiry + origy;

					tminz[0] = ((swizzledLoz >> 0) & 0xFF) * adjusted_idirz + origz;
					tminz[1] = ((swizzledLoz >> 8) & 0xFF) * adjusted_idirz + origz;
					tminz[2] = ((swizzledLoz >> 16) & 0xFF) * adjusted_idirz + origz;
					tminz[3] = ((swizzledLoz >> 24) & 0xFF) * adjusted_idirz + origz;

					tmaxx[0] = ((swizzledHix >> 0) & 0xFF) * adjusted_idirx + origx;
					tmaxx[1] = ((swizzledHix >> 8) & 0xFF) * adjusted_idirx + origx;
					tmaxx[2] = ((swizzledHix >> 16) & 0xFF) * adjusted_idirx + origx;
					tmaxx[3] = ((swizzledHix >> 24) & 0xFF) * adjusted_idirx + origx;

					tmaxy[0] = ((swizzledHiy >> 0) & 0xFF) * adjusted_idiry + origy;
					tmaxy[1] = ((swizzledHiy >> 8) & 0xFF) * adjusted_idiry + origy;
					tmaxy[2] = ((swizzledHiy >> 16) & 0xFF) * adjusted_idiry + origy;
					tmaxy[3] = ((swizzledHiy >> 24) & 0xFF) * adjusted_idiry + origy;

					tmaxz[0] = ((swizzledHiz >> 0) & 0xFF) * adjusted_idirz + origz;
					tmaxz[1] = ((swizzledHiz >> 8) & 0xFF) * adjusted_idirz + origz;
					tmaxz[2] = ((swizzledHiz >> 16) & 0xFF) * adjusted_idirz + origz;
					tmaxz[3] = ((swizzledHiz >> 24) & 0xFF) * adjusted_idirz + origz;

					for (int childIndex = 0; childIndex < 4; childIndex++)
					{
						// Use VMIN, VMAX to compute the slabs
						const float cmin = fmaxf(fmax_fmax(tminx[childIndex], tminy[childIndex], tminz[childIndex]), tmin);
						const float cmax = fminf(fmin_fmin(tmaxx[childIndex], tmaxy[childIndex], tmaxz[childIndex]), tmax);

						bool intersected = cmin <= cmax;

						// Potential micro-optimization: use VSHL to implement this part, as described by the paper
						if (intersected)
						{
							const unsigned int child_bits = extract_byte(child_bits4, childIndex);
							const unsigned int bit_index = extract_byte(bit_index4, childIndex);
							hitmask |= child_bits << bit_index;
						}
					}
				}

				nodeGroup.y = (hitmask & 0xFF000000) | (*((byte*)&n0.w + 3));
				triangleGroup.y = hitmask & 0x00FFFFFF;
			}
		}
		else
		{
			triangleGroup = nodeGroup;
			nodeGroup = make_uint2(0);
		}

#if TRIANGLE_POSTPONING
		const int totalThreads = __popc(__activemask());
#endif

		while (triangleGroup.y != 0)
		{
#if TRIANGLE_POSTPONING
			const float Rt = 0.2;
			const int threshold = totalThreads * Rt;
			const int numActiveThreads = __popc(__activemask());
			if (numActiveThreads < threshold)
			{
				traversalStack[stackPtr++] = triangleGroup;
				break;
			}
#endif

			int triIdx = __bfind(triangleGroup.y);

			int triAddr = triangleGroup.x * 3 + triIdx * 3;

			float4 v00 = __ldg(TriangleWoopCoordinates + triAddr + 0);
			float4 v11 = __ldg(TriangleWoopCoordinates + triAddr + 1);
			float4 v22 = __ldg(TriangleWoopCoordinates + triAddr + 2);

			float Oz = v00.w - orig.x * v00.x - orig.y * v00.y - orig.z * v00.z;
			float invDz = 1.0f / (dir.x * v00.x + dir.y * v00.y + dir.z * v00.z);
			float t = Oz * invDz;

			float Ox = v11.w + orig.x * v11.x + orig.y * v11.y + orig.z * v11.z;
			float Dx = dir.x * v11.x + dir.y * v11.y + dir.z * v11.z;
			float u = Ox + t * Dx;

			float Oy = v22.w + orig.x * v22.x + orig.y * v22.y + orig.z * v22.z;
			float Dy = dir.x * v22.x + dir.y * v22.y + dir.z * v22.z;
			float v = Oy + t * Dy;

			if (t > tmin && t < tmax)
			{
				if (u >= 0.0f && u <= 1.0f)
				{
					if (v >= 0.0f && u + v <= 1.0f)
					{
						int triangleIndex = __ldg(MappingFromTriangleAddressToIndex + triAddr);
						int materialIndex = __ldg(MappingFromTriangleAddressToIndex + triAddr + 1);
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
							if (IsShadowRay || FORCE_SHADOWRAYS)
							{
								OutHitInfo.TriangleIndex = 0;
								return;
							}
							else
							{
								trianglenormal = cross(make_float3(v22.x, v22.y, v22.z), make_float3(v11.x, v11.y, v11.z));
								triangleuv.x = u;
								triangleuv.y = v;

								tmax = t;
								hitAddr = triAddr;
							}
						}
					}
				}
			}

			triangleGroup.y &= ~(1 << triIdx);
		}

		if (hitAddr != -1 && (IsShadowRay || FORCE_SHADOWRAYS))
		{
			OutHitInfo.TriangleIndex = 0;
			return;
		}

		if (nodeGroup.y <= 0x00FFFFFF)
		{
			if (stackPtr > 0)
			{
				nodeGroup = traversalStack[--stackPtr];
			}
			else
			{
				break;
			}
		}
	} while (true);

	OutHitInfo.TriangleUV = triangleuv;
	OutHitInfo.TriangleNormalUnnormalized = trianglenormal;
	OutHitInfo.TriangleIndex = hitAddr != -1 ? __ldg(MappingFromTriangleAddressToIndex + hitAddr) : -1;
	OutHitInfo.HitDistance = tmax;
}
