#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "global_const.h"

__global__ void update_scene_kernel(float* dPositions, float* dVelocities,
	float* dRadius, float* dElasticities, float dt, const unsigned int NUM_OBJECT);

__global__ void init_cells_kernel(unsigned int* cells, unsigned int* objects,
	float* positions, float* radius, unsigned int* cellCount,
	const unsigned int NUM_OBJECT, const float MAX_DIM);

__global__ void radix_tabulate_kernel(unsigned int* keys, unsigned int* radices,
	unsigned int cellsPerGroup, int shift, unsigned int n);

__global__ void radix_sum_kernel(unsigned int* radices, unsigned int* radixSums);

__global__ void radix_order_kernel(unsigned int* keysIn, unsigned int* valuesIn,
	unsigned int* keysOut, unsigned int* valuesOut,
	unsigned int* radices, unsigned int* radixSums,
	unsigned int cellsPerGroup, int shift, unsigned int n);

__global__ void cells_collide_kernel(unsigned int* cells, unsigned int* objects,
	float* positions, float* velocities, float* radius,
	unsigned int numCells, unsigned int cellsPerThread,
	unsigned int* collisionCnt, unsigned int* testCnt,
	unsigned int* collisionMatrix, const unsigned int NUM_OBJECT);

__global__ void set_new_p_and_v_kernel(float* positions, float* velocities,
	float* radius, float* elasticities, float* masses,
	unsigned int* collisionMatrix, float* newPositions, float* newVelocities,
	const unsigned int NUM_OBJECT);