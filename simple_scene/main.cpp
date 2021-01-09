#define _CRT_SECURE_NO_WARNINGS
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <Windows.h>

#include "Camera.h"
#include "Scene.h"
#include "Shader.h"
#include "Object.h"

// set window
//const unsigned int SCR_WIDTH = 800;
//const unsigned int SCR_HEIGHT = 600;

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

// set cameare
Camera camera;

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// set time
float deltaTime = 0.0f;
float lastFrame = 0.0f;


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

GLFWwindow* init_GLFW();

void test(const int NUM) {
	const int MODE = 2;
	Scene scene(MODE, NUM);
	scene.test();
}

void show_animation(const int MODE) {
	// initialization
	GLFWwindow* window = init_GLFW();
	if (!window) {
		glfwTerminate();
		exit(-1);
	}

	int numObject, len;
	float maxRadius = 0.5;
	if (MODE == 0) {
		numObject = 1000;
		len = 10;
	}
	else {
		numObject = 27;
		len = 3;
	}

	// set scene
	Scene scene(MODE, numObject, maxRadius, len);
	scene.set_vertices_data();

	// set camera
	camera.set(glm::vec3(16.0f, 16.0f, 16.0f), glm::vec3(0.0f, 1.0f, 0.0f), -135.0f, -45.0f);
	lastX = SCR_WIDTH / 2.0f;
	lastY = SCR_HEIGHT / 2.0f;
	firstMouse = true;

	// set shader light
	Shader shader("shader.vs", "shader.fs");
	shader.use();

	shader.setVec3("light.position", scene.lightPos);
	shader.setVec3("light.ambient", scene.ambientColor);
	shader.setVec3("light.diffuse", scene.diffuseColor);
	shader.setVec3("light.specular", scene.specularStrength);
	shader.setFloat("light.constant", scene.constant);
	shader.setFloat("light.linear", scene.linear);
	shader.setFloat("light.quadratic", scene.quadratic);

	// render loop

	lastFrame = glfwGetTime();
	while (!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		// input
		processInput(window);

		// render
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		// view and projection
		shader.use();
		shader.setVec3("viewPos", camera.Position);
		glm::mat4 projection = glm::perspective(
			glm::radians(camera.Zoom), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera.get_view_matrix();
		shader.setMat4("projection", projection);
		shader.setMat4("view", view);

		scene.draw(shader);
		// update and draw objects in scene
		scene.update(deltaTime);
		//Sleep(100);
		//lastFrame += 0.1;

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
}

int main() {
	while (true) {
		system("cls");
		cout << "Please Select:\n"
			<< "0. Animation_0\n"
			<< "1. Animation_1\n"
			<< "2. Test\n" << endl;
		string in;
		cin >> in;
		cout << endl;
		if (in == "0") {
			show_animation(0);
		}
		else if (in == "1") {
			show_animation(1);
		}
		else if (in == "2") {
			while (true) {
				system("cls");
				cout << "Please Input Number of the Balls:\n" << endl;
				int num;
				cin >> num;
				cout << "Number of the Balls is: " << num << endl << endl;
				if (num > 0 && num <= 10000) {
					test(num);
					cout << "Press Any Key to Go Back" << endl;
					system("pause");
					continue;
				}
			}
		}
	}
	return 0;
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.process_keyboard(FORWARD, deltaTime * 2);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.process_keyboard(BACKWARD, deltaTime * 2);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.process_keyboard(LEFT, deltaTime * 2);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.process_keyboard(RIGHT, deltaTime * 2);
	}
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		camera.process_keyboard(UP, deltaTime * 2);
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		camera.process_keyboard(DOWN, deltaTime * 2);
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.process_mouse_movement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.process_mouse_scroll(yoffset);
}

GLFWwindow* init_GLFW() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Ball Collision", NULL, NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return nullptr;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return nullptr;
	}
	glEnable(GL_DEPTH_TEST);
	return window;
}

