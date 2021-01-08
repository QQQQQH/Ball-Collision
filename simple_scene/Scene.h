#pragma once
#include <iostream>
#include <random>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "collision_compute.h"
#include "Object.h"
#include "Shader.h"

using namespace std;

class Scene {
	static const int COLOR_NUM = 10;
	static const float color[COLOR_NUM][3];

	const unsigned int
		LEN,
		NUM_OBJECT,
		OBJECT_SIZE,
		CELL_SIZE;

	const float
		MIN_RADIUS,
		MAX_RADIUS,
		MAX_DIM;

	const int MODE;

	// host
	unsigned int
		numCells,
		cntCollisions,
		cntTests,
		* collisionMatrix;

	float
		* positions,
		* velocities,
		* radius,
		* elasticities,
		* masses;

	int* colorNums;

	// device
	float
		* dPositions,
		* dVelocities,
		* dRadius,
		* dElasticities,
		* dMasses,

		* dNewPositions,
		* dNewVelocities;

	unsigned int
		* dTemp,
		* dCollisionMatrix;

	unsigned int
		* dCells,
		* dCellsTmp,
		* dObjects,
		* dObjectsTmp,
		* dRadices,
		* dRadixSums;


	Material pureColorMaterial;

	Sphere* balls;
	Plane* floorPlane;

	void set_scene();

	void device_malloc();
	void copy_to_device();
	void copy_to_host();
	void update_scene_on_device(float dt);

	void init_cells();
	void sort_cells();
	void cells_collide();
	void set_new_p_and_v();

	void get_cnt_collision();
	void naive_collide();
public:
	glm::vec3
		lightPos = glm::vec3(5.0f, 2.0f, 5.0f),
		lightColor = glm::vec3(1.0f, 1.0f, 1.0f),
		diffuseColor = lightColor * glm::vec3(0.8f),
		ambientColor = lightColor * glm::vec3(0.4f),
		specularStrength = glm::vec3(1.0f, 1.0f, 1.0f);
	float
		constant = 1.0f,
		linear = 0.014f,
		quadratic = 0.0007f;

	Scene(const int MODE = 0, const unsigned int NUM_OBJECT0 = 27,
		const float MAX_RADIUS0 = 0.5, const unsigned int LEN = 3);
	void set_vertices_data();
	void update(float dt);
	void draw(Shader& shader);

	void test();
};

