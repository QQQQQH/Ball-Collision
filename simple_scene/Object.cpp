#include "Object.h"

Sphere* Sphere::sphere = nullptr;

const float Plane::vertices[36] = {
	// top
	-1.0f, 0.0f,  1.0f, 0.0f, 1.0f, 0.0f,
	 1.0f, 0.0f,  1.0f, 0.0f, 1.0f, 0.0f,
	 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	-1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	-1.0f, 0.0f,  1.0f, 0.0f, 1.0f, 0.0f,
};

void Plane::set_vertices_data() {
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(0);
	// normal attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}

void Plane::draw() {
	glBindVertexArray(VAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
}

Sphere::Sphere(const Material& material0) : Object(material0) {
	int idx = 0;

	for (int y = 0; y <= Y_SEGMENTS; ++y) {
		for (int x = 0; x <= X_SEGMENTS; x++) {
			float xSegment = (float) x / (float) X_SEGMENTS;
			float ySegment = (float) y / (float) Y_SEGMENTS;
			float xPos = cos(xSegment * 2.0f * PI) * sin(ySegment * PI);
			float yPos = cos(ySegment * PI);
			float zPos = sin(xSegment * 2.0f * PI) * sin(ySegment * PI);
			vertices[idx++] = xPos;
			vertices[idx++] = yPos;
			vertices[idx++] = zPos;
		}
	}

	idx = 0;
	for (int i = 0; i < Y_SEGMENTS; ++i) {
		for (int j = 0; j < X_SEGMENTS; j++) {
			indices[idx++] = i * (X_SEGMENTS + 1) + j;
			indices[idx++] = (i + 1) * (X_SEGMENTS + 1) + j;
			indices[idx++] = (i + 1) * (X_SEGMENTS + 1) + j + 1;
			indices[idx++] = i * (X_SEGMENTS + 1) + j;
			indices[idx++] = (i + 1) * (X_SEGMENTS + 1) + j + 1;
			indices[idx++] = i * (X_SEGMENTS + 1) + j + 1;
		}
	}
}

Sphere* Sphere::get_sphere(const Material& material0) {
	if (sphere) {
		return sphere;
	}
	sphere = new Sphere(material0);
	return sphere;
}

void Sphere::set_vertices_data() {
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, VERTICES_NUM * sizeof(float), &vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, INDEICES_NUM * sizeof(int), &indices[0], GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*) 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Sphere::prepare_draw() {
	glBindVertexArray(VAO);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void Sphere::draw() {
	glDrawElements(GL_TRIANGLES, INDEICES_NUM, GL_UNSIGNED_INT, 0);
}
