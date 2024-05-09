#pragma once

#define DYNAMIC_FETCH 1
#define TRIANGLE_POSTPONING 1

__device__ unsigned __bfind(unsigned i) { unsigned b; asm volatile("bfind.u32 %0, %1; " : "=r"(b) : "r"(i)); return b; }

__device__ __inline__ uint sign_extend_s8x4(uint i) { uint v; asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(v) : "r"(i)); return v; }

__device__ __inline__ uint extract_byte(uint i, uint n) { return (i >> (n * 8)) & 0xFF; }
