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

const unsigned int
NUM_BLOCKS = 100,
THREADS_PER_BLOCK = 512;