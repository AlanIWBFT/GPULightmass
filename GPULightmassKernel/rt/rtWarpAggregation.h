const int WARP_SZ = 32;

__device__ inline int lane_id(void) { return (threadIdx.y * blockDim.x + threadIdx.x) % WARP_SZ; }

__device__ int warp_bcast(int v, int leader) { return __shfl_sync(__activemask(), v, leader); }

// warp-aggregated atomic increment
__device__ int atomicAggInc(int *ctr) {
	int mask = __ballot_sync(__activemask(), 1);
	// select the leader
	int leader = __ffs(mask) - 1;
	// leader does the update
	int res;
	if (lane_id() == leader)
		res = atomicAdd(ctr, __popc(mask));
	// broadcast result
	res = warp_bcast(res, leader);
	// each thread computes its own value
	return res + __popc(mask & ((1 << lane_id()) - 1));
} // atomicAggInc
