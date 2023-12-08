//OPTIMIZE AND CHECK SO I USE SHARED MEMORY PROPERLY


#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
__shared__ unsigned int localBins[NUM_BINS];

int index = blockIdx.x * blockDim.x + threadIdx.x;

// Initialize localBins in shared memory to zeros
    if (threadIdx.x < num_bins) {
        localBins[threadIdx.x] = 0;
    }

    __syncthreads();

//Add counter for localbin that corresponds with input value
if(index < num_elements)
{
  atomicAdd(&localBins[threadIdx.x], 1);
}

__syncthreads();

if(threadIdx.x < num_bins)
{
  atomicAdd(&bins[input[index]], localBins[threadIdx.x]);
}

}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
int index = blockIdx.x * blockDim.x + threadIdx.x;

if(index < num_bins && bins[index] > 127)
{
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

  //@@ Insert code below to read in inputLength from args
  inputLength = std::atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocHost(&hostInput, inputLength*sizeof(unsigned int));
  cudaMallocHost(&hostBins, NUM_BINS*sizeof(unsigned int));
  cudaMallocHost(&resultRef, inputLength*sizeof(unsigned int));

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for(int i = 0; i < inputLength; i++)
  {
    hostInput[i] = rand() % NUM_BINS;
  }


  //@@ Insert code below to create reference result in CPU
  //initialize result to zero
  for(int i = 0; i < NUM_BINS; i++)
  {
    resultRef[i] = 0;
  }

  //Histogram, and also saturize if resultref is 127
  for(int i = 0; i < inputLength; i++)
  {
    if(resultRef[hostInput[i]] < 127)
    {
      resultRef[hostInput[i]] += 1;
    }
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput,inputLength*sizeof(unsigned int));
  cudaMalloc(&deviceBins,NUM_BINS*sizeof(unsigned int));


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, inputLength*sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  //For the first kernel we want to launch the same amount of threads as inputLength:
  int blockSize(256);
  int gridSize((inputLength+blockSize -1)/blockSize);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<dim3(gridSize), dim3(blockSize)>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Initialize the second grid and block dimensions here
  //For second kernel we want to launch one kernel for each bin:
  int convertBlockSize(256);
  int convertGridSize((NUM_BINS+convertBlockSize -1)/convertBlockSize);


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<dim3(convertGridSize), dim3(convertBlockSize)>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  int resultCounter = 0;
  for(int i = 0; i < NUM_BINS; i++)
  {
    if(resultRef[i] == hostBins[i])
    {
      resultCounter++;
    }else{
      printf("Reference Bin: %d\n Device Bin: %d\n index: %d\n", resultRef[i], hostBins[i], i);
    }
  }
  printf("Amount correct: %d/%d", resultCounter, NUM_BINS);

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);


  //@@ Free the CPU memory here
  cudaFreeHost(hostInput);
  cudaFreeHost(hostBins);
  cudaFreeHost(resultRef);

  return 0;
}

