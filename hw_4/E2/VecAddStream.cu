#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len, int offset) {
  //@@ Insert code to implement vector addition here

  //Have to know index of kernel that is running
  int index = offset + blockIdx.x*blockDim.x + threadIdx.x;

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

  int nStreams = 4;
  const int streamSize = (inputLength)/nStreams;
  const int streamBytes = streamSize * sizeof(DataType);

  cudaStream_t stream[nStreams+1];
  for(int i = 0; i < nStreams+1; ++i)
  {
    cudaStreamCreate(&stream[i]);
  }

  startTime = start_timer();

  for(int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
  cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);

  }


  //@@ Initialize the 1D grid and block dimensions here
  int blockSize = 512;

  for(int i = 0; i < nStreams; ++i)
  {
  int offset = i*streamSize;
  //@@ Launch the GPU Kernel here
  vecAdd<<<(streamSize+blockSize-1)/blockSize, blockSize, 0, stream[i]>>>(deviceInput1, deviceInput2, deviceOutput, inputLength, offset);
    cudaDeviceSynchronize(); //Make sure kernel is finished before proceeding
  }


  for(int i = 0; i < nStreams; ++i)
  {
    int offset = i*streamSize;
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost,stream[i]);
  }

  for(int i = 0; i < nStreams+1; ++i)
  {
    cudaStreamDestroy(stream[i]);
  }

  endTime = stop_timer();
  elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
  printf("Elapsed Time: %.6f seconds\n", elapsedTime);
  

  //@@ Insert code below to compare the output with the reference
  int amountCorrect = 0;

  for(int i = 0; i < inputLength; ++i)
  {
    if(resultRef[i] == hostOutput[i])
    {
      amountCorrect += 1;
    }else
    {
      printf("Wrong!\n Reference: %lf\n Device: %lf\n", resultRef[i], hostOutput[i]);
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