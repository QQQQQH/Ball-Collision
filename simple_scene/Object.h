#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Material.h"

#define PI 3.14159265358979323846

class Object {
public:
	Material material;
	Object(const Material& material0) :material(material0) {}
	glm::vec3 ambient() const { return material.ambient; }
	glm::vec3 diffuse() const { return material.diffuse; }
	glm::vec3 specular() const { return material.specular; }
	float shininess() const { return material.shininess; }
};

class Plane : public Object {
	static const float vertices[36];
	unsigned int VAO, VBO;

public:
	Plane(const Material& material0) : Object(material0) {}

	void set_vertices_data();

	void draw();
};

class Sphere : public Object {
	static Sphere* sphere;
	static const int
		Y_SEGMENTS = 50,
		X_SEGMENTS = 50,
		VERTICES_NUM = (X_SEGMENTS + 1) * (Y_SEGMENTS + 1) * 3,
		INDEICES_NUM = X_SEGMENTS * Y_SEGMENTS * 6;
	unsigned int VAO, VBO, EBO;
	float vertices[VERTICES_NUM];
	int indices[INDEICES_NUM];

public:
	Sphere(const Material& material0);

	static Sphere* get_sphere(const Material& material0);

	void set_vertices_data();

	void prepare_draw();

	void draw();
};


