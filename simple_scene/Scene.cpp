#include "Scene.h"

__constant__ float
G = 9.8,
EPS = 1e-5;
__constant__ float PLANE[3][2] = {
	0,10,
	0,10000,
	0,10
};

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

Scene::Scene(const unsigned int NUM_OBJECT0, const float MAX_RADIUS0) :
	NUM_OBJECT(NUM_OBJECT0),
	OBJECT_SIZE(3 * NUM_OBJECT0 * sizeof(float)),
	CELL_SIZE(8 * NUM_OBJECT0 * sizeof(unsigned int)),
	MAX_RADIUS(MAX_RADIUS0), MIN_RADIUS(0.6 * MAX_RADIUS0), MAX_DIM(8 * MAX_RADIUS0) {

	positions = (float*) malloc(OBJECT_SIZE);
	velocities = (float*) malloc(OBJECT_SIZE);
	radius = (float*) malloc(OBJECT_SIZE);
	elasticities = (float*) malloc(OBJECT_SIZE);
	masses = (float*) malloc(OBJECT_SIZE);
	colorNums = (int*) malloc(OBJECT_SIZE);

	device_malloc();
	set_scene();

	pureColorMaterial.set_pure_color();
	balls = Sphere::get_sphere(pureColorMaterial);
	floorPlane = new Plane(pureColorMaterial);
	floorPlane->material.set_color(glm::vec3(1, 1, 1));
}

void Scene::set_vertices_data() {
	balls->set_vertices_data();
	floorPlane->set_vertices_data();
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
		positions[i * 3] = i % LEN + 4;
		positions[i * 3 + 1] = 3 + i / (LEN * LEN);
		positions[i * 3 + 2] = (i / LEN) % LEN + 4;

		// velocity
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

	copy_to_device();
}

void Scene::device_malloc() {
	cudaMalloc((void**) &dPositions, OBJECT_SIZE);
	cudaMalloc((void**) &dVelocities, OBJECT_SIZE);
	cudaMalloc((void**) &dRadius, OBJECT_SIZE);
	cudaMalloc((void**) &dElasticities, OBJECT_SIZE);
	cudaMalloc((void**) &dMasses, OBJECT_SIZE);

	cudaMalloc((void**) &dNewPositions, OBJECT_SIZE);
	cudaMalloc((void**) &dNewVelocities, OBJECT_SIZE);

	cudaMalloc((void**) &dTemp, 2 * sizeof(unsigned int));
	cudaMalloc((void**) &dCells, CELL_SIZE);
	cudaMalloc((void**) &dCellsTmp, CELL_SIZE);
	cudaMalloc((void**) &dObjects, CELL_SIZE);
	cudaMalloc((void**) &dObjectsTmp, CELL_SIZE);
	cudaMalloc((void**) &dRadices, NUM_BLOCKS_SORT * GROUPS_PER_BLOCK * NUM_RADICES * sizeof(unsigned int));
	cudaMalloc((void**) &dRadixSums, NUM_RADICES * sizeof(unsigned int));

	cudaMalloc((void**) &dCollisionMatrix, NUM_OBJECT * NUM_OBJECT * sizeof(unsigned int));

	collisionMatrix = (unsigned int*) malloc(NUM_OBJECT * NUM_OBJECT * sizeof(unsigned int));
}

void Scene::copy_to_device() {
	cudaMemcpy(dPositions, positions, OBJECT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dVelocities, velocities, OBJECT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dRadius, radius, OBJECT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dElasticities, elasticities, OBJECT_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dMasses, masses, OBJECT_SIZE, cudaMemcpyHostToDevice);
}

void Scene::copy_to_host() {
	cudaMemcpy(positions, dPositions, OBJECT_SIZE, cudaMemcpyDeviceToHost);
}

void Scene::update_scene_on_device(float dt) {
	// update positions and velocities
	update_scene_kernel << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (
		dPositions, dVelocities, dRadius, dElasticities, dt, NUM_OBJECT);
	cudaDeviceSynchronize();

	// detect collision
	init_cells();
	sort_cells();
	cells_collide();

	// if collisions, update positions and velocities caused by collisions
	if (cntCollisions) {
		set_new_p_and_v();
	}
}

void Scene::init_cells() {
	cudaMemset(dTemp, 0, sizeof(unsigned int));
	init_cells_kernel << <NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(unsigned int) >> > (
		dCells, dObjects, dPositions, dRadius, dTemp, NUM_OBJECT, MAX_DIM);
	cudaMemcpy(&numCells, dTemp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

void Scene::sort_cells() {
	unsigned int cellsPerGroup = (NUM_OBJECT * DIM_2 - 1) / NUM_BLOCKS_SORT / GROUPS_PER_BLOCK + 1;
	unsigned int
		* cellsSwap,
		* objectsSwap;

	// stable sort
	for (int i = 0; i < 32; i += L) {
		radix_tabulate_kernel << <NUM_BLOCKS_SORT, GROUPS_PER_BLOCK* THREADS_PER_GROUP,
			GROUPS_PER_BLOCK* NUM_RADICES * sizeof(unsigned int) >> > (
				dCells, dRadices, cellsPerGroup, i, NUM_OBJECT * DIM_2);
		radix_sum_kernel << <NUM_BLOCKS_SORT, GROUPS_PER_BLOCK* THREADS_PER_GROUP,
			PADDED_GROUPS * sizeof(unsigned int) >> > (dRadices, dRadixSums);
		radix_order_kernel << <NUM_BLOCKS_SORT, GROUPS_PER_BLOCK* THREADS_PER_GROUP,
			NUM_RADICES * sizeof(unsigned int) + GROUPS_PER_BLOCK *
			NUM_RADICES * sizeof(unsigned int) >> > (
				dCells, dObjects, dCellsTmp, dObjectsTmp, dRadices, dRadixSums,
				cellsPerGroup, i, NUM_OBJECT * DIM_2);

		// swap
		cellsSwap = dCells;
		dCells = dCellsTmp;
		dCellsTmp = cellsSwap;
		objectsSwap = dObjects;
		dObjects = dObjectsTmp;
		dObjectsTmp = objectsSwap;
	}

}

// get collided objects and store them in the collisionMatrix
void Scene::cells_collide() {

	unsigned int cellsPerThread = (numCells - 1) / NUM_BLOCKS /
		THREADS_PER_BLOCK + 1;

	cudaMemset(dTemp, 0, 2 * sizeof(unsigned int));
	cudaMemset(dCollisionMatrix, 0, NUM_OBJECT * NUM_OBJECT * sizeof(unsigned int));
	cells_collide_kernel << <NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(unsigned int) >> > (
		dCells, dObjects, dPositions, dVelocities, dRadius, numCells,
		cellsPerThread, dTemp, dTemp + 1, dCollisionMatrix, NUM_OBJECT);
	cudaMemcpy(&cntCollisions, dTemp, sizeof(unsigned int),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(&cntTests, dTemp + 1, sizeof(unsigned int),
		cudaMemcpyDeviceToHost);
}

// set new position and new velocity caused by colliding
void Scene::set_new_p_and_v() {
	set_new_p_and_v_kernel << <NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK* DIM_P_AND_V * sizeof(float) >> >
		(dPositions, dVelocities, dRadius, dElasticities, dMasses,
			dCollisionMatrix, dNewPositions, dNewVelocities, NUM_OBJECT);

	cudaMemcpy(dPositions, dNewPositions, OBJECT_SIZE, cudaMemcpyDeviceToDevice);
	cudaMemcpy(dVelocities, dNewVelocities, OBJECT_SIZE, cudaMemcpyDeviceToDevice);
}

void Scene::test() {}

void Scene::update(float dt) {
	update_scene_on_device(dt);
	copy_to_host();
}

void Scene::draw(Shader& shader) {
	shader.use();

	// draw floorPlane
	glm::mat4 model(1.0f);
	model = glm::translate(model, glm::vec3(5, 0, 5));
	model = glm::scale(model, glm::vec3(5));
	shader.setMat4("model", model);

	shader.setVec3("material.ambient", floorPlane->ambient());
	shader.setVec3("material.diffuse", floorPlane->diffuse());
	shader.setVec3("material.specular", floorPlane->specular());
	shader.setFloat("material.shininess", floorPlane->shininess());
	floorPlane->draw();

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
