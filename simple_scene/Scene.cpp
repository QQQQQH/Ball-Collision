#include "Scene.h"

const float Scene::color[COLOR_NUM][3] = {
	0.13725490196078433, 0.803921568627451, 0.7137254901960784,
	0.5568627450980392, 0.8980392156862745, 0.8352941176470589,
	0.5803921568627451, 0.984313725490196, 0.9411764705882353,
	0.7529411764705882, 0.9686274509803922, 0.9882352941176471,
	0.6745098039215687, 0.9725490196078431, 0.8274509803921568,
	0.9019607843137255, 0.8941176470588236, 0.7529411764705882,
	0.9058823529411765, 0.792156862745098, 0.4980392156862745,
	1.0, 0.8431372549019608, 0.7333333333333333,
	1.0, 0.7019607843137254, 0.6588235294117647,
	1.0, 0.2980392156862745, 0.4235294117647059
};

Scene::Scene() {

	positions = (float*) malloc(OBJECT_SIZE);
	velocities = (float*) malloc(OBJECT_SIZE);
	radius = (float*) malloc(OBJECT_SIZE);
	elasticities = (float*) malloc(OBJECT_SIZE);
	masses = (float*) malloc(OBJECT_SIZE);
	colorNums = (int*) malloc(OBJECT_SIZE);

	device_malloc(NUM_OBJECT);
	set_scene();

	pureColorMaterial.set_pure_color();
	balls = Sphere::get_sphere(pureColorMaterial);
	floor = new Plane(pureColorMaterial);
	floor->material.set_color(glm::vec3(1, 1, 1));
}

void Scene::set_vertices_data() {
	balls->set_vertices_data();
	floor->set_vertices_data();
}

void Scene::set_scene() {
	const int MAX_VELOCITY = 50;
	const float
		MIN_RADIUS = 0.3,
		MAX_RADIUS = 0.5;

	//int flag = 0;
	int flag = 1;

	// set position, velocity, radius, elasticity, mass and color of the balls
	for (int i = 0; i < NUM_OBJECT; ++i) {
		// position
		positions[i * 3] = i % LEN + 0.5;
		positions[i * 3 + 1] = 3 + i / (LEN * LEN);
		positions[i * 3 + 2] = (i / LEN) % LEN + 0.5;

		velocities[i * 3 + 1] = 0;

		if (flag) {
			// velocity
			velocities[i * 3] = -MAX_VELOCITY + (float) (rand() % 100) / 100 * 2 * MAX_VELOCITY;
			velocities[i * 3 + 2] = -MAX_VELOCITY + (float) (rand() % 100) / 100 * 2 * MAX_VELOCITY;
			//printf("%d %f %f %f\n", i, velocities[i * DIM], velocities[i * DIM + 1], velocities[i * DIM + 2]);

			// radius
			radius[i] = MIN_RADIUS + (MAX_RADIUS - MIN_RADIUS) * (float) (rand() % 100) / 100;

			// elasticity
			elasticities[i] = 0.8 + (float) (rand() % 100) / 1000;
		}
		else {
			// velocity
			velocities[i * 3] = 0;
			velocities[i * 3 + 2] = 0;

			// radius
			radius[i] = MIN_RADIUS;

			// elasticity
			elasticities[i] = 0.8;

		}

		// mass
		masses[i] = (radius[i] * 10) * (radius[i] * 10);

		// color
		colorNums[i] = i % COLOR_NUM;
	}

	copy_to_device(positions, velocities, radius, elasticities, masses);
}

void Scene::update(float dt) {
	update_scene_on_device(dt);
	copy_to_host(positions);
}

void Scene::draw(Shader& shader) {
	shader.use();

	// draw floor
	glm::mat4 model(1.0f);
	model = glm::translate(model, glm::vec3(5, 0, 5));
	model = glm::scale(model, glm::vec3(5));
	shader.setMat4("model", model);

	shader.setVec3("material.ambient", floor->ambient());
	shader.setVec3("material.diffuse", floor->diffuse());
	shader.setVec3("material.specular", floor->specular());
	shader.setFloat("material.shininess", floor->shininess());
	floor->draw();

	// draw balls
	balls->prepare_draw();

	for (int i = 0; i < NUM_OBJECT; i++) {
		shader.use();

		glm::mat4 model(1.0f);
		model = glm::translate(model, glm::vec3(
			positions[i * 3],
			positions[i * 3 + 1],
			positions[i * 3 + 2]));
		model = glm::scale(model, glm::vec3(radius[i]));
		shader.setMat4("model", model);

		int colorNum = colorNums[i];
		balls->material.set_color(glm::vec3(
			color[colorNum][0],
			color[colorNum][1],
			color[colorNum][2]));
		shader.setVec3("material.ambient", balls->ambient());
		shader.setVec3("material.diffuse", balls->diffuse());
		shader.setVec3("material.specular", balls->specular());
		shader.setFloat("material.shininess", balls->shininess());

		balls->draw();
	}
}
