#pragma once

const unsigned int
L = 8,
NUM_RADICES = 256,
NUM_BLOCKS_SORT = 16,
GROUPS_PER_BLOCK = 12,
THREADS_PER_GROUP = 16,
PADDED_BLOCKS = 16,
PADDED_GROUPS = 256;

const unsigned int
DIM = 3,
DIM_2 = 8,
DIM_3 = 27,
DIM_P_AND_V = 6;

const int
LOW_BOUND = 0,
HIGH_BOUND = 20;

const unsigned int LEN = 3;
//const unsigned int NUM_OBJECT = 45;
const unsigned int NUM_OBJECT = LEN * LEN * LEN;
const float MAX_DIM = 2.5;

const unsigned int
OBJECT_SIZE = 3 * NUM_OBJECT * sizeof(float),
CELL_SIZE = 8 * NUM_OBJECT * sizeof(unsigned int),
NUM_BLOCKS = 100,
THREADS_PER_BLOCK = 512;

// host
static unsigned int
numCells,
cntCollisions,
cntTests,
* collisionMatrix;

// device
static float
* dPositions,
* dVelocities,
* dRadius,
* dElasticities,
* dMasses,

* dNewPositions,
* dNewVelocities;

static unsigned int
* dTemp,
* dCollisionMatrix;

static unsigned int
* dCells,
* dCellsTmp,
* dObjects,
* dObjectsTmp,
* dRadices,
* dRadixSums;
