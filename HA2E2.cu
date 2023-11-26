
#include <stdio.h>
#include <sys/time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here

  //Find out index of thread that are doing the multiplication:
  int row = blockIdx.y*blockDim.y+threadIdx.y;

  int column = blockIdx.x*blockDim.x+threadIdx.x;

  //Check so that index does not go out of range (In matrix mult Arow is larger or equal to Brow and Bcolumn is larger or equal to Acolumn)
  if(row < numARows && column < numBColumns)
  {
    C[row * numBColumns + column] = 0.0;
    for(int i = 0; i < numAColumns; i++)
    {
      C[row * numBColumns + column] += A[numAColumns * row + i]*B[i*numBColumns+column];
    }
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

  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
  numARows = std::atoi(argv[1]);
  numAColumns = std::atoi(argv[2]);
  numBRows = std::atoi(argv[3]);
  numBColumns = std::atoi(argv[4]);

  //Make sure that size works

  numCRows = numARows;
  numCColumns = numBColumns;

  if(numAColumns == numBRows)
  {
    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    //@@ Insert code below to allocate Host memory for input and output
    cudaMallocHost(&hostA, numARows*numAColumns*sizeof(DataType));
    cudaMallocHost(&hostB, numBRows*numBColumns*sizeof(DataType));
    cudaMallocHost(&hostC, numCRows*numCColumns*sizeof(DataType));
    cudaMallocHost(&resultRef, numCRows*numCColumns*sizeof(DataType));


    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    for(int i = 0; i < numARows*numAColumns; i++)
    {
      hostA[i] = rand() % (10+1);
    }
    printf("\n");
    for(int i = 0; i < numBRows*numBColumns; i++)
    {
      hostB[i] = rand()% (10+1);
    }

    for(int i = 0; i < numARows; i++)
    {
      for(int j = 0; j < numBColumns; j++)
      {
        for(int k = 0; k < numAColumns; k++)
        {
          resultRef[i*numCColumns+j] += hostA[i*numAColumns+k]*hostB[k*numBColumns+j];
        }
      }
    }

    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceA, numARows*numAColumns*sizeof(DataType));
    cudaMalloc(&deviceB, numBRows*numBColumns*sizeof(DataType));
    cudaMalloc(&deviceC, numCRows*numCColumns*sizeof(DataType));


    startTime = start_timer();
    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    endTime = stop_timer();
    elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Copy Host to Device elapsed Time: %.6f seconds\n", elapsedTime);


    //@@ Initialize the grid and block dimensions here
    int blockSize = 32;

    int gridSizeX = (blockSize + numCColumns -1) /blockSize;
    int gridSizeY = (blockSize + numCRows -1) /blockSize;

    dim3 gridSize(gridSizeX, gridSizeY);
    dim3 blockSize3(blockSize,blockSize);

    startTime = start_timer();
    //@@ Launch the GPU Kernel here
    gemm<<<gridSize, blockSize3>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    endTime = stop_timer();
    elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Kernel elapsed Time: %.6f seconds\n", elapsedTime);


    startTime = start_timer();
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(DataType), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    endTime = stop_timer();
    elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Device to Host elapsed Time: %.6f seconds\n", elapsedTime);


    //@@ Insert code below to compare the output with the reference
    int amountCorrect = 0;
    for(int i = 0; i < numCRows*numCColumns; i++)
    {
      if(resultRef[i] == hostC[i])
      {
        amountCorrect += 1;
      }
    }

    printf("Amount correct: %d/%d \n", amountCorrect, numCRows*numCColumns);

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);


    //@@ Free the CPU memory here
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFreeHost(hostC);
    cudaFreeHost(resultRef);

  } else
  printf("Error, cannot do matrix multiplication with those values, (columns in A has to match rows in B) \n");
 


  return 0;
}
