#include <stdio.h>
#include <curand_kernel.h>

__device__ void switchSpin(int *grid, int *cols, int i, int j) {
    grid[i*(*cols) + j] = -1 * grid[i*(*cols) + j];
}

__device__ void calculateEnergy(int *grid, int *cols, int i, int j, int *energy) {
    *energy = -grid[i*(*cols) + j] * (grid[(i+1)*(*cols) + j] + grid[(i-1)*(*cols) + j] + grid[i*(*cols) + (j+1)] + grid[i*(*cols) + (j-1)]);
}

extern "C"
__global__ void runIsingModel(int *grid, int *rows, int *cols, double *temp, int *steps) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    curandState_t rng;
	curand_init(clock64(), index, 0, &rng);

    int i, j, energy;
    double r;
    
    for (int k = index; k < *steps; k+=stride) {
        i = curand_uniform_double(&rng) * (*rows-2) + 1;
        i = (int)(i+0.5);
        j = curand_uniform_double(&rng) * (*cols-2) + 1;
        j = (int)(j+0.5);

        calculateEnergy(grid, cols, i, j, &energy);

        if (energy > 0) {
            switchSpin(grid, cols, i, j);
        }
        else if (energy < 0) {
            r = curand_uniform_double(&rng);
            if (r < exp(2*energy/(*temp))) {
                switchSpin(grid, cols, i, j);
            }
        }
    }
}