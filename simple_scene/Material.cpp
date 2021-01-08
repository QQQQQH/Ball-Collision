#include "Material.h"

void Material::set_color(const glm::vec3& color) {
	ambient = color;
	diffuse = color;
	specular = glm::vec3(0.5f) * color;
	shininess = 32.0f;
}

void Material::set_pure_color() {
	kShade = 1.0f;
	kReflect = 0.0f;
	kRefract = 0.0f;
}

void Material::set_metal() {
	kShade = 0.9f;
	kReflect = 0.1f;
	kRefract = 0.0f;
}

void Material::set_smooth() {
	kShade = 0.5f;
	kReflect = 0.5f;
	kRefract = 0.0f;
}

void Material::set_full_reflect() {
	kShade = 0.0f;
	kReflect = 1.0f;
	kRefract = 0.0f;
}

void Material::set_full_refract() {
	kShade = 0.0f;
	kReflect = 0.0f;
	kRefract = 1.0f;
}

void Material::set_whit_plastic() {
	ambient = glm::vec3(0.0);
	diffuse = glm::vec3(0.55);
	specular = glm::vec3(0.7);
	shininess = 0.25 * 128;
}

void Material::set_cyan_rubber() {
	ambient = glm::vec3(0.0, 0.05, 0.05);
	diffuse = glm::vec3(0.4, 0.5, 0.5);
	specular = glm::vec3(0.04, 0.7, 0.7);
	shininess = 0.078125 * 128;
}

void Material::set_jade() {
	ambient = glm::vec3(0.135, 0.2225, 0.1575);
	diffuse = glm::vec3(0.54, 0.89, 0.63);
	specular = glm::vec3(0.316228, 0.316228, 0.316228);
	shininess = 0.1 * 128;
}
