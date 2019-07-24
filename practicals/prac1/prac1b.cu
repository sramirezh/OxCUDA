//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;
}

__global__ void VecAdd(float* A,float *B,float *C) 
{
int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

//
// main code
//

int main(int argc, const char **argv)
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 
  
//For question 7
  float *h_A,*h_B,*h_C,*d_A,*d_B,*d_C;
  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per blocksd

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  
  h_x = (float *)malloc(nsize*sizeof(float));
  
checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));
  
//allocatae memory for the vectors

h_A = (float *)malloc(nsize*sizeof(float));
h_B = (float *)malloc(nsize*sizeof(float));
h_C = (float *)malloc(nsize*sizeof(float));

checkCudaErrors(cudaMalloc((void **)&d_A, nsize*sizeof(float)));
checkCudaErrors(cudaMalloc((void **)&d_B, nsize*sizeof(float)));
checkCudaErrors(cudaMalloc((void **)&d_C, nsize*sizeof(float)));  
	
  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>(d_x);
  getLastCudaError("my_first_kernel execution failed\n");

// copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,nsize*sizeof(float),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(d_x));
  free(h_x);

///#################

// execute VecAdd Kernel

// Sending from host to device, notice that C was initialised therefore there is no need to send as it is going to be
// rewrited

checkCudaErrors( cudaMemcpy(d_A,h_A,nsize*sizeof(float),cudaMemcpyHostToDevice));
checkCudaErrors( cudaMemcpy(d_B,h_B,nsize*sizeof(float),cudaMemcpyHostToDevice));
VecAdd<<<nblocks,nthreads>>>(d_A,d_B,d_C);
 
// Sending from device to host

checkCudaErrors( cudaMemcpy(h_C,d_C,nsize*sizeof(float),cudaMemcpyDeviceToHost));

// Now printing the Array
for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_C[n]);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
