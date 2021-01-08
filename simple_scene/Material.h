#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Material {
public:
	glm::vec3
		ambient = glm::vec3(1.0f),
		diffuse = ambient,
		specular = glm::vec3(0.5f) * ambient;
	float
		shininess = 32.0f,
		kShade = 1.0f,
		kReflect = 0.0f,
		kRefract = 0.0f,
		refractiveIndex = 1.52f;

	Material(const glm::vec3& color = glm::vec3(1.0f)) :
		ambient(color), diffuse(color), specular(color* glm::vec3(0.5f)) {}

	void set_color(const glm::vec3& color);
	void set_pure_color();
	void set_metal();
	void set_smooth();
	void set_full_reflect();
	void set_full_refract();
	void set_whit_plastic();
	void set_cyan_rubber();
	void set_jade();
};

