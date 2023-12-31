#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here

  //Have to know index of kernel that is running
  int index = blockIdx.x*blockDim.x + threadIdx.x;

  if(index < len)
  {
    out[index] = in1[index] + in2[index];
  }
}

//@@ Insert code to implement timer start
clock_t start_timer()
{
  return clock();
}


//@@ Insert code to implement timer stop
clock_t stop_timer()
{
  return clock();
}

int main(int argc, char **argv) {
  clock_t startTime;
  clock_t endTime;
  double elapsedTime;

  startTime = start_timer();

  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;
  //@@ Insert code below to read in inputLength from args
  //we assume first line in command are
  inputLength = std::atoi(argv[1]);


  //printf("The input length is %d\n", inputLength);




  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
  cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
  cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));
  cudaMallocHost(&resultRef, inputLength * sizeof(DataType));

  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for(int i = 0; i < inputLength; i++)
  {
    hostInput1[i] = rand();
    hostInput2[i] = rand();
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));


  startTime = start_timer();
  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  //@@ Initialize the 1D grid and block dimensions here
  int blockSize = 1024;

  //ensures full data set is covered even if inputlength is not a multiple of blocksize
  int gridSize = (inputLength + blockSize - 1) / blockSize;

 
  //@@ Launch the GPU Kernel here
  vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

  cudaDeviceSynchronize(); //Make sure kernel is finished before proceeding

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength*sizeof(DataType), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  endTime = stop_timer();
  elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
  printf("Elapsed Time: %.6f seconds\n", elapsedTime);

  //@@ Insert code below to compare the output with the reference
  int amountCorrect = 0;

  for(int i = 0; i < inputLength; i++)
  {
    if(resultRef[i] == hostOutput[i])
    {
      amountCorrect += 1;
    }
  }
  printf("amount correct: %d / %d \n", amountCorrect, inputLength);

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  cudaFreeHost(resultRef);


  return 0;
}