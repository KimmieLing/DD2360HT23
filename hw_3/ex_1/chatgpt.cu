#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    __shared__ unsigned int localBins[NUM_BINS];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize localBins in shared memory to zeros
    if (threadIdx.x < num_bins) {
        localBins[threadIdx.x] = 0;
    }

    __syncthreads();

    // Compute histogram using shared memory and atomics
    if (index < num_elements) {
        atomicAdd(&localBins[threadIdx.x], 1);
    }

    __syncthreads();

    // Copy results from shared memory to global memory using atomics
    if (threadIdx.x < num_bins) {
        atomicAdd(&bins[input[index]], localBins[threadIdx.x]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Clean up bins that saturate at 127
    if (index < num_bins && bins[index] > 127) {
        bins[index] = 127;
    }
}

int main(int argc, char **argv) {
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    // Read in inputLength from args
    inputLength = std::atoi(argv[1]);

    printf("The input length is %d\n", inputLength);

    // Allocate Host memory for input and output
    cudaMallocHost(&hostInput, inputLength * sizeof(unsigned int));
    cudaMallocHost(&hostBins, NUM_BINS * sizeof(unsigned int));
    cudaMallocHost(&resultRef, NUM_BINS * sizeof(unsigned int));

    // Initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = rand() % NUM_BINS;
    }

    // Create reference result in CPU
    for (int i = 0; i < NUM_BINS; i++) {
        resultRef[i] = 0;
    }

    // Histogram, and also saturate if resultRef is 127
    for (int i = 0; i < inputLength; i++) {
        if (resultRef[hostInput[i]] < 128) {
            resultRef[hostInput[i]] += 1;
        }
    }

    // Allocate GPU memory
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    // Copy memory to the GPU
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    // Initialize the grid and block dimensions
    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;

    // Launch the GPU Kernel for histogram
    histogram_kernel<<<gridSize, blockSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    cudaDeviceSynchronize();

    // Initialize the second grid and block dimensions
    int convertBlockSize = 256;
    int convertGridSize = (NUM_BINS + convertBlockSize - 1) / convertBlockSize;

    // Launch the second GPU Kernel for saturation cleanup
    convert_kernel<<<convertGridSize, convertBlockSize>>>(deviceBins, NUM_BINS);
    cudaDeviceSynchronize();

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Compare the output with the reference
    int resultCounter = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (resultRef[i] == hostBins[i]) {
            resultCounter++;
        }
    }
    printf("Amount correct: %d/%d\n", resultCounter, NUM_BINS);

    // Free the GPU memory
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    // Free the CPU memory
    cudaFreeHost(hostInput);
    cudaFreeHost(hostBins);
    cudaFreeHost(resultRef);

    return 0;
}