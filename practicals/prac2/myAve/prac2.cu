
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float a,b,c;


////////////////////////////////////////////////////////////////////////
// kernel routinGe
////////////////////////////////////////////////////////////////////////


__global__ void function(float *d_z,float *d_value)
{
  int   ind;

  // move array pointers to correct position

  ind = threadIdx.x +blockIdx.x*blockDim.x;

  // function calculation
	
  d_value[ind]=a*d_z[ind]*d_z[ind]+b*d_z[ind]+c;


//   printf(" ind, d_z, d_value =  %d %f %f \n",ind,d_z[ind],d_value[ind]); 
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int Npoints=640000;	
  float h_a,h_b,h_c; 
  float   *h_value, *d_value, *d_z;
  double  sum1;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_value = (float *)malloc(sizeof(float)*Npoints);

  checkCudaErrors( cudaMalloc((void **)&d_value, sizeof(float)*Npoints) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*Npoints) );

  // define constants and transfer to GPU

  h_a     = 1.0f;
  h_b     = 2.0f;
  h_c     = 3.0f;

  checkCudaErrors( cudaMemcpyToSymbol(a,    &h_a,    sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(b,    &h_b,    sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(c,    &h_c,    sizeof(h_c)) );

  // random number generation

  cudaEventRecord(start);

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateNormal(gen, d_z,Npoints, 0.0f, 1.0f) );
 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, Npoints/(0.001*milli));

  // execute kernel and time it

  cudaEventRecord(start);

  function<<<Npoints/64, 64>>>(d_z, d_value);
  getLastCudaError("function execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Averaging execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_value, d_value, sizeof(float)*Npoints,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  for (int i=0; i<Npoints; i++) {
    sum1 += h_value[i];
  }

  printf("\nAverage value   = %13.8f\n\n",sum1/Npoints);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_value);
  checkCudaErrors( cudaFree(d_value) );
  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
