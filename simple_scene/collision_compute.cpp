#include <iostream>
#include "collision_compute.h"

__constant__ float
G = 9.8,
EPS = 1e-6;
__constant__ float PLANE[3][2] = {
	0,10,
	0,10000,
	0,10
};

__device__ void d_sum_reduce(unsigned int* values, unsigned int* out) {
	// wait for the whole array to be populated
	__syncthreads();

	// sum by reduction, using half the threads in each subsequent iteration
	unsigned int threads = blockDim.x;
	unsigned int half = threads / 2;

	while (half) {
		if (threadIdx.x < half) {
			for (int k = threadIdx.x + half; k < threads; k += half) {
				values[threadIdx.x] += values[k];
			}

			threads = half;
		}

		half /= 2;

		// make sure all the threads are on the same iteration
		__syncthreads();
	}

	// only let one thread update the current sum
	if (!threadIdx.x) {
		atomicAdd(out, values[0]);
	}
}

__device__ void d_sum_reduce(float* values, float* out, unsigned int dim = 1) {
	// wait for the whole array to be populated
	__syncthreads();

	// sum by reduction, using half the threads in each subsequent iteration
	unsigned int threads = blockDim.x;
	unsigned int half = threads / 2;

	while (half) {
		if (threadIdx.x < half) {
			for (int i = threadIdx.x + half; i < threads; i += half) {
				for (int j = 0; j < dim; ++j) {
					values[threadIdx.x * dim + j] += values[i * dim + j];
				}
			}
			threads = half;
		}
		half /= 2;
		// make sure all the threads are on the same iteration
		__syncthreads();
	}

	// only let one thread update the current sum
	if (!threadIdx.x && values != out) {
		for (int i = 0; i < dim; ++i) {
			atomicAdd(out + i, values[i]);
		}
	}
}

__device__ void d_prefix_sum(unsigned int* values, unsigned int n) {
	int offset = 1;
	int a;
	unsigned int temp;

	// upsweep
	for (int d = n / 2; d; d /= 2) {
		__syncthreads();

		if (threadIdx.x < d) {
			a = (threadIdx.x * 2 + 1) * offset - 1;
			values[a + offset] += values[a];
		}

		offset *= 2;
	}

	if (!threadIdx.x) {
		values[n - 1] = 0;
	}

	// downsweep
	for (int d = 1; d < n; d *= 2) {
		__syncthreads();
		offset /= 2;

		if (threadIdx.x < d) {
			a = (threadIdx.x * 2 + 1) * offset - 1;
			temp = values[a];
			values[a] = values[a + offset];
			values[a + offset] += temp;
		}
	}
}

// calculate new position through dt and velocity
__global__ void update_scene_kernel(float* dPositions, float* dVelocities,
	float* dRadius, float* dElasticities, float dt, const unsigned int NUM_OBJECT) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < NUM_OBJECT; i += blockDim.x * gridDim.x) {
		bool hitFloor = false;
		for (int j = 0; j < DIM; ++j) {
			// new position = old position + v * t
			dPositions[i * DIM + j] += dVelocities[i * DIM + j] * dt;

			// hit left/down plane
			if (dPositions[i * DIM + j] - dRadius[i] - PLANE[j][0] < EPS) {
				if (j == 1) {
					hitFloor = true;
				}
				dPositions[i * DIM + j] = PLANE[j][0] + dRadius[i] + EPS;
				dVelocities[i * DIM + j] = -dVelocities[i * DIM + j] * dElasticities[i];
			}
			if (j == 1) {
				continue;
			}

			// hit right/up plane
			if (PLANE[j][1] - dPositions[i * DIM + j] - dRadius[i] < EPS) {
				dPositions[i * DIM + j] = PLANE[j][1] - dRadius[i] - EPS;
				dVelocities[i * DIM + j] = -dVelocities[i * DIM + j] * dElasticities[i];
			}
		}

		// gravity
		//if (!hitFloor) {
		//	dVelocities[i * DIM + 1] -= G * dt;
		//}
		dVelocities[i * DIM + 1] -= G * dt;
	}
}

__global__ void init_cells_kernel(unsigned int* cells, unsigned int* objects,
	float* positions, float* radius, unsigned int* cellCount,
	const unsigned int NUM_OBJECT, const float MAX_DIM) {
	extern __shared__ unsigned int t[];
	unsigned int count = 0;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < NUM_OBJECT; i += gridDim.x * blockDim.x) {
		unsigned int hash = 0;
		unsigned int sides = 0;
		int h = i * DIM_2;
		int m = 1;
		int q;
		int r;
		float x;
		float a;

		// find home cell
		for (int j = 0; j < DIM; ++j) {
			x = positions[i * DIM + j];

			// cell ID hash
			hash = (hash << 8) | (unsigned int) (x / MAX_DIM);

			x -= floor(x / MAX_DIM) * MAX_DIM;
			a = radius[i];
			sides <<= 2;

			//  sides stores which side of the center the object overlaps
			if (x < a) {
				sides |= 3;
			}
			else if (MAX_DIM - x < a) {
				sides |= 1;
			}
		}

		// home cell
		cells[h] = (hash << 1) | 0x00;
		objects[h] = (i << 1) | 0x01;
		++count;

		// find phantom dCells in the Moore neighborhood
		for (int j = 0; j < DIM_3; ++j) {
			// skip the home cell
			if (j == DIM_3 / 2) {
				continue;
			}

			q = j;
			hash = 0;

			for (int k = 0; k < DIM; ++k) {
				r = q % 3 - 1;
				x = positions[i * DIM + k];

				// skip this cell if the object is on the wrong side
				if (r && (sides >> (DIM - k - 1) * 2 & 0x03 ^ r) & 0x03 ||
					x + r * MAX_DIM <= LOW_BOUND ||
					x + r * MAX_DIM >= HIGH_BOUND) {
					hash = UINT32_MAX;
					break;
				}

				hash = hash << 8 | (unsigned int) (x / MAX_DIM) + r;
				q /= 3;
			}

			if (hash != UINT32_MAX) {
				++count;
				++h;

				//phantom cell
				cells[h] = hash << 1 | 0x01;
				objects[h] = i << 1 | 0x00;

				// number of dCells occupied
				++m;
			}
		}

		// fill up remaining dCells
		while (m < DIM_2) {
			++h;
			cells[h] = UINT32_MAX;
			objects[h] = i << 2;
			++m;
		}
	}

	// perform reduction to count number of dCells occupied
	t[threadIdx.x] = count;
	d_sum_reduce(t, cellCount);
}

__global__ void radix_tabulate_kernel(unsigned int* keys, unsigned int* radices,
	unsigned int cellsPerGroup, int shift, unsigned int n) {


	extern __shared__ unsigned int s[];
	int group = threadIdx.x / THREADS_PER_GROUP;
	int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cellsPerGroup;
	int group_end = group_start + cellsPerGroup;
	unsigned int k;

	// initialize shared memory
	for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i +=
		blockDim.x) {
		s[i] = 0;
	}

	__syncthreads();

	// count instances of each radix
	for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end &&
		i < n; i += THREADS_PER_GROUP) {
		// need only avoid bank conflicts by group
		k = (keys[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;

		// increment radix counters sequentially by thread in the thread group
		for (int j = 0; j < THREADS_PER_GROUP; ++j) {
			if (threadIdx.x % THREADS_PER_GROUP == j) {
				++s[k];
			}
		}
	}

	__syncthreads();

	// copy to global memory
	for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i +=
		blockDim.x) {
		radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS_SORT + blockIdx.x) *
			GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] = s[i];
	}
}

__global__ void radix_sum_kernel(unsigned int* radices, unsigned int* radixSums) {
	extern __shared__ unsigned int s[];
	unsigned int total;
	unsigned int left = 0;
	unsigned int* radix = radices + blockIdx.x * NUM_RADICES * GROUPS_PER_BLOCK;

	for (int j = 0; j < NUM_RADICES / NUM_BLOCKS_SORT; ++j) {
		// initialize shared memory
		for (int i = threadIdx.x; i < NUM_BLOCKS_SORT * GROUPS_PER_BLOCK; i +=
			blockDim.x) {
			s[i] = radix[i];
		}

		__syncthreads();

		// add padding to array for prefix-sum
		for (int i = threadIdx.x + NUM_BLOCKS_SORT * GROUPS_PER_BLOCK; i <
			PADDED_GROUPS; i += blockDim.x) {
			s[i] = 0;
		}

		__syncthreads();

		if (!threadIdx.x) {
			total = s[PADDED_GROUPS - 1];
		}

		// calculate prefix-sum on radix counters
		d_prefix_sum(s, PADDED_GROUPS);
		__syncthreads();

		// copy to global memory
		for (int i = threadIdx.x; i < NUM_BLOCKS_SORT * GROUPS_PER_BLOCK; i +=
			blockDim.x) {
			radix[i] = s[i];
		}

		__syncthreads();

		// calculate total sum and copy to global memory
		if (!threadIdx.x) {
			total += s[PADDED_GROUPS - 1];

			// calculate prefix-sum on local radices
			radixSums[blockIdx.x * NUM_RADICES / NUM_BLOCKS_SORT + j] = left;
			total += left;
			left = total;
		}

		// move to next radix
		radix += NUM_BLOCKS_SORT * GROUPS_PER_BLOCK;
	}
}

__global__ void radix_order_kernel(unsigned int* keysIn, unsigned int* valuesIn,
	unsigned int* keysOut, unsigned int* valuesOut,
	unsigned int* radices, unsigned int* radixSums,
	unsigned int cellsPerGroup, int shift, unsigned int n) {

	extern __shared__ unsigned int s[];
	unsigned int* t = s + NUM_RADICES;
	int group = threadIdx.x / THREADS_PER_GROUP;
	int group_start = (blockIdx.x * GROUPS_PER_BLOCK + group) * cellsPerGroup;
	int group_end = group_start + cellsPerGroup;
	unsigned int k;

	// initialize shared memory
	for (int i = threadIdx.x; i < NUM_RADICES; i += blockDim.x) {
		s[i] = radixSums[i];

		// copy the last element in each prefix-sum to a separate array
		if (!((i + 1) % (NUM_RADICES / NUM_BLOCKS_SORT))) {
			t[i / (NUM_RADICES / NUM_BLOCKS_SORT)] = s[i];
		}
	}

	__syncthreads();

	// add padding to array for prefix-sum
	for (int i = threadIdx.x + NUM_BLOCKS_SORT; i < PADDED_BLOCKS; i += blockDim.x) {
		t[i] = 0;
	}

	__syncthreads();

	// calculate prefix-sum on radix counters
	d_prefix_sum(t, PADDED_BLOCKS);
	__syncthreads();

	// add offsets to prefix-sum values
	for (int i = threadIdx.x; i < NUM_RADICES; i += blockDim.x) {
		s[i] += t[i / (NUM_RADICES / NUM_BLOCKS_SORT)];
	}

	__syncthreads();

	// add offsets to radix counters
	for (int i = threadIdx.x; i < GROUPS_PER_BLOCK * NUM_RADICES; i +=
		blockDim.x) {
		t[i] = radices[(i / GROUPS_PER_BLOCK * NUM_BLOCKS_SORT + blockIdx.x) *
			GROUPS_PER_BLOCK + i % GROUPS_PER_BLOCK] + s[i / GROUPS_PER_BLOCK];
	}

	__syncthreads();

	// rearrange key-value pairs
	for (int i = group_start + threadIdx.x % THREADS_PER_GROUP; i < group_end &&
		i < n; i += THREADS_PER_GROUP) {
		// need only avoid bank conflicts by group
		k = (keysIn[i] >> shift & NUM_RADICES - 1) * GROUPS_PER_BLOCK + group;

		// write key-value pairs sequentially by thread in the thread group
		for (int j = 0; j < THREADS_PER_GROUP; ++j) {
			if (threadIdx.x % THREADS_PER_GROUP == j) {
				keysOut[t[k]] = keysIn[i];
				valuesOut[t[k]] = valuesIn[i];
				++t[k];
			}
		}
	}
}

__global__ void cells_collide_kernel(unsigned int* cells, unsigned int* objects,
	float* positions, float* velocities,
	float* radius, unsigned int numCells,
	unsigned int cellsPerThread,
	unsigned int* collisionCnt, unsigned int* testCnt,
	unsigned int* collisionMatrix, const unsigned int NUM_OBJECT) {

	extern __shared__ unsigned int t[];

	int
		threadStart = (blockIdx.x * blockDim.x + threadIdx.x) * cellsPerThread,
		threadEnd = threadStart + cellsPerThread,
		start = -1,
		i = threadStart;
	unsigned int last = UINT32_MAX;
	unsigned int
		numH,
		numP,
		cntCollisions = 0,
		cntTests = 0;


	while (1) {
		// find cell ID change indices
		if (i >= numCells || cells[i] >> 1 != last) {
			// at least one home-cell object and at least one other object present
			if (start + 1 && numH >= 1 && numH + numP >= 2) {
				for (int j = start; j < start + numH; ++j) {
					unsigned int home = objects[j] >> 1;
					float dh = radius[home];

					for (int k = j + 1; k < i; ++k) {
						// count the number of tests performed
						++cntTests;
						unsigned int phantom = objects[k] >> 1;
						float dp = radius[phantom] + dh;
						float d = 0, dx;

						for (int l = 0; l < DIM; ++l) {
							dx = positions[phantom * DIM + l] - positions[home * DIM + l];
							d += dx * dx;
						}

						// if collision
						if (d < dp * dp - EPS) {
							++cntCollisions;
							collisionMatrix[home * NUM_OBJECT + phantom] = collisionMatrix[phantom * NUM_OBJECT + home] = 1;
						}
					}
				}
			}

			if (i > threadEnd || i >= numCells) {
				break;
			}

			// the first thread starts immediately; the others wait until a change
			if (i != threadStart || !blockIdx.x && !threadIdx.x) {
				// reset counters for new cell
				numH = 0;
				numP = 0;
				start = i;
			}

			last = cells[i] >> 1;
		}

		if (start + 1) {
			if (objects[i] & 0x01) {
				// home dCells
				++numH;
			}
			else {
				// phantom dCells
				++numP;
			}
		}

		++i;
	}

	t[threadIdx.x] = cntCollisions;
	d_sum_reduce(t, collisionCnt);

	__syncthreads();

	t[threadIdx.x] = cntTests;
	d_sum_reduce(t, testCnt);
}

__global__ void set_new_p_and_v_kernel(float* positions, float* velocities,
	float* radius, float* elasticities, float* masses,
	unsigned int* collisionMatrix, float* newPositions, float* newVelocities,
	const unsigned int NUM_OBJECT) {

	extern __shared__ float newPosAndVel[];

	for (int i = blockIdx.x; i < NUM_OBJECT; i += gridDim.x) {

		// initialize shared memory
		for (int k = 0; k < DIM_P_AND_V; ++k) {
			newPosAndVel[threadIdx.x * DIM_P_AND_V + k] = 0;
		}

		for (int j = threadIdx.x; j < NUM_OBJECT; j += blockDim.x) {
			if (collisionMatrix[i * NUM_OBJECT + j]) {
				// calculate new position
				if (i > j) {
					float d = 0, dx;
					for (int k = 0; k < DIM; ++k) {
						dx = positions[i * DIM + k] - positions[j * DIM + k];
						d += dx * dx;
					}
					d = sqrt(d);
					float dp = radius[i] + radius[j];
					float delta = (dp - d + EPS) / 2;
					float ratio = (d + delta) / d;

					for (int k = 0; k < DIM; ++k) {
						newPosAndVel[threadIdx.x * DIM_P_AND_V + k] +=
							(positions[j * DIM + k] + (positions[i * DIM + k] - positions[j * DIM + k]) * ratio);
					}
				}
				else {
					for (int k = 0; k < DIM; ++k) {
						newPosAndVel[threadIdx.x * DIM_P_AND_V + k] += positions[i * DIM + k];
					}
				}

				// calculate new velocity
				for (int k = 0; k < DIM; ++k) {
					float v1 = velocities[i * DIM + k], v2 = velocities[j * DIM + k],
						m1 = masses[i], m2 = masses[j];
					newPosAndVel[threadIdx.x * DIM_P_AND_V + DIM + k] +=
						(v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2);
				}
			}
		}

		d_sum_reduce(newPosAndVel, newPosAndVel, DIM_P_AND_V);

		// set new position and velocity caused by colliding in thread 0
		if (threadIdx.x == 0) {
			unsigned int cntCollisions = 0;

			for (int k = 0; k < NUM_OBJECT; ++k) {
				if (collisionMatrix[i * NUM_OBJECT + k]) {
					++cntCollisions;
				}
			}

			if (cntCollisions) {
				for (int k = 0; k < DIM; ++k) {
					newPositions[i * DIM + k] = newPosAndVel[k] / cntCollisions;
					newVelocities[i * DIM + k] = newPosAndVel[DIM + k] / cntCollisions * elasticities[i];
				}
			}
			else {
				for (int k = 0; k < DIM; ++k) {
					newPositions[i * DIM + k] = positions[i * DIM + k];
					newVelocities[i * DIM + k] = velocities[i * DIM + k];
				}
			}
			//if (abs(newVelocities[i * DIM + 2]) <= 0.1) {
			//printf("--- %d %f %f %f\NUM_OBJECT", i, newVelocities[i * DIM], newVelocities[i * DIM + 1], newVelocities[i * DIM + 2]);
			//}
		}

		__syncthreads();
	}
}