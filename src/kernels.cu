#include <curand_kernel.h>
typedef unsigned long long ull;

__global__ void valueGeneratorKernel(float *randomNumbers, int N, ull seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < N){
        curandState state;
        curand_init(seed, idx, 0, &state);
        randomNumbers[idx] = curand_uniform(&state) + static_cast<float>(threadIdx.x) * 0.0001f;
    }
}

extern "C" void generateRandomVectors(float *randomNumbers, int n, int dimm, ull seed, cudaStream_t stream) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n*dimm + threadsPerBlock - 1)/threadsPerBlock;
    valueGeneratorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(randomNumbers, n*dimm, seed);
}

__global__ void dotProductKernel(float *vectorsBuffer, float* sampleVector, float* productsBuffer, int vecNum, int vecDimm){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < vecNum){
        float dotProduct = 0.0;
        for (int i = 0; i < vecDimm; i++) dotProduct += vectorsBuffer[idx*vecDimm+i]*sampleVector[i];
        productsBuffer[idx] = dotProduct;
    }
}

extern "C" void generateDotProducts(float *vectorsBuffer, float* sampleVector, float* productsBuffer, int vecNum, int vecDimm, cudaStream_t stream){
    int threadsPerBlock = 1024;
    int blocksPerGrid = (vecNum + threadsPerBlock - 1)/threadsPerBlock;

    dotProductKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(vectorsBuffer, sampleVector, productsBuffer, vecNum, vecDimm);
}