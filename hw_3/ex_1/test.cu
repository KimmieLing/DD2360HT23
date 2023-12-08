#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

    __shared__ unsigned int localBins[NUM_BINS];
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize localBins to 0
    if (threadIdx.x < NUM_BINS) {
        localBins[threadIdx.x] = 0;
    }
    __syncthreads();

    // Compute histogram using atomicAdd
    if (index < num_elements) {
        atomicAdd(&localBins[input[index]], 1);
    }
    __syncthreads();

    // Update global bins using atomicAdd
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&bins[threadIdx.x], localBins[threadIdx.x]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert bins to 127 if they saturate
    if (tid < num_bins && bins[tid] > 127) {
        bins[tid] = 127;
    }
}

int main(int argc, char **argv) {
  
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    //@@ Insert code below to read in inputLength from args
    inputLength = std::atoi(argv[1]);

    printf("The input length is %d\n", inputLength);
    
    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocHost(&hostInput, inputLength * sizeof(unsigned int));
    cudaMallocHost(&hostBins, NUM_BINS * sizeof(unsigned int));
    cudaMallocHost(&resultRef, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = rand() % NUM_BINS;
    }

    //@@ Insert code below to create reference result in CPU
    for (int i = 0; i < NUM_BINS; i++) {
        resultRef[i] = 0;
    }
    for (int i = 0; i < inputLength; i++) {
        resultRef[hostInput[i]]++;
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    //@@ Insert code to Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //@@ Insert code to initialize GPU results
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    //@@ Initialize the grid and block dimensions here
    dim3 blockDim(256);
    dim3 gridDim((inputLength + blockDim.x - 1) / blockDim.x);

    //@@ Launch the GPU Kernel here
    histogram_kernel<<<gridDim, blockDim>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

    //@@ Initialize the second grid and block dimensions here
    dim3 convertBlockDim(256);
    dim3 convertGridDim((NUM_BINS + convertBlockDim.x - 1) / convertBlockDim.x);

    //@@ Launch the second GPU Kernel here
    convert_kernel<<<convertGridDim, convertBlockDim>>>(deviceBins, NUM_BINS);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //@@ Insert code below to compare the output with the reference
    for (int i = 0; i < NUM_BINS; i++) {
        if (hostBins[i] != resultRef[i]) {
            printf("Test failed!\n");
            break;
        }
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    //@@ Free the CPU memory here
    cudaFreeHost(hostInput);
    cudaFreeHost(hostBins);
    cudaFreeHost(resultRef);

    return 0;
}