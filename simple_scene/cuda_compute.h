#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_global.h"

void device_malloc(unsigned int numObjects);

void copy_to_device(float* positions, float* velocities, float* radius, float* elasticities, float* masses);

void copy_to_host(float* positions);

void update_scene_on_device(float dt);

