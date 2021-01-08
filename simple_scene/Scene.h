#pragma once
#include <iostream>
#include <random>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cuda_compute.h"
#include "Object.h"
#include "Shader.h"

using namespace std;

class Scene {
	static const int COLOR_NUM = 10;
	static const float color[COLOR_NUM][3];
	float
		* positions,
		* velocities,
		* radius,
		* elasticities,
		* masses;
	int* colorNums;

	Material pureColorMaterial;

	Sphere* balls;
	Plane* floor;

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

	Scene();
	void set_vertices_data();
	void set_scene();
	void update(float dt);
	void draw(Shader& shader);
};

