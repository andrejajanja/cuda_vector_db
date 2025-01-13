#include <iostream>
#include <string>
#include <random>
#include <chrono>

#include <cuda_runtime.h>

typedef unsigned long long ull;

extern "C" void generateRandomVectors(float *randomNumbers, int n, int dimm, ull seed, cudaStream_t stream);
extern "C" void generateDotProducts(float *vectorsBuffer, float* sampleVector, float* productsBuffer, int vecNum, int vecDimm, cudaStream_t stream);

inline float generateRandomNumberCPU(int i, ull seed){
    ull state = (((ull)i + seed) * 6364136223846793005ULL + 1) % (1ULL << 48);
    return static_cast<float>(state & 0xFFFFFF) / (1 << 24);
}

void cpuMain(int vecNum, int vecDimm){
    int vectorsFootprint = vecNum*vecDimm*sizeof(float);

    float* vectors = (float*)malloc(vectorsFootprint);

    auto now = std::chrono::system_clock::now();
    ull seed = (ull)std::chrono::system_clock::to_time_t(now);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < vecNum*vecDimm; i++){
        
        vectors[i] = generateRandomNumberCPU(i, seed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[CPU] Vector generation => " << duration << "ms" << std::endl;

    free(vectors);
}

void gpuMain(int vecNum, int vecDimm){
    std::cout << "[GPU] Init + random vector generation\n";
    int vectorsFootprint = vecNum*vecDimm*sizeof(float);
    int vecFootprint = vecDimm*sizeof(float);
    int dotFootprint = vecNum*sizeof(float);

    float* h_sampleVectorBuffer = (float*)malloc(vecFootprint);
    float *h_dotProductsBuffer = (float*)malloc(dotFootprint);

    auto now = std::chrono::system_clock::now();
    ull seed = (ull)std::chrono::system_clock::to_time_t(now);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::random_device rd;  // Seed for the random number engine
    std::mt19937 eng(rd());  // Mersenne Twister engine
    std::uniform_real_distribution<float> distr(0.0, 1.0);  // Define the range

    auto start = std::chrono::high_resolution_clock::now();
    
    float *d_vectorsBuffer, *d_dotProductsBuffer, *d_sampleVectorBuffer;
    cudaMalloc(&d_vectorsBuffer, vectorsFootprint);

    generateRandomVectors(d_vectorsBuffer, vecNum, vecDimm, seed, stream);
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "[GPU] Vector generation => " << duration << "ms\n\n";
    
    
    cudaMalloc(&d_dotProductsBuffer, dotFootprint);
    cudaMalloc(&d_sampleVectorBuffer, vecFootprint);

    std::cout << "Input 1 to generate random vector and find index of the vector that has highest dot product. Everything other than 1 stops the execution\n\n";

    int inputStatus;
    //add loop here for vec gen and shit
    while(true){
        std::cin>>inputStatus;
        std::cout<<"\033[A";

        if(inputStatus != 1){
            break;
        }

        now = std::chrono::system_clock::now();
        seed = (ull)std::chrono::system_clock::to_time_t(now);

        for (int i = 0; i < vecDimm; i++) h_sampleVectorBuffer[i] = distr(eng);

        start = std::chrono::high_resolution_clock::now();    

        cudaMemcpyAsync(d_sampleVectorBuffer, h_sampleVectorBuffer, vecFootprint, cudaMemcpyHostToDevice, stream);

        generateDotProducts(d_vectorsBuffer, d_sampleVectorBuffer, d_dotProductsBuffer, vecNum, vecDimm, stream);
    
        cudaMemcpyAsync(h_dotProductsBuffer, d_dotProductsBuffer, dotFootprint, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        int maxIndex = 0;
        float maxProduct = 0.0;

        for (int i = 0; i < vecNum; i++){
            if(h_dotProductsBuffer[i] > maxProduct){
                maxIndex = i;
                maxProduct = h_dotProductsBuffer[i];
            }
        }
        
        std::cout<<"[GPU] "<<maxIndex<<"/"<<vecNum<<" => "<<duration<<"ms"<<std::endl;
    }


    cudaFreeHost(h_sampleVectorBuffer);
    cudaFreeHost(h_dotProductsBuffer);

    cudaStreamDestroy(stream);

    cudaFree(d_vectorsBuffer);
    cudaFree(d_dotProductsBuffer);
    cudaFree(d_sampleVectorBuffer);
}


int main(int argc, char* argv[]) {
    if(argc != 4) {
        std::cout<<"Invalid number of command line arguments - 2 uints are required and a path to a file\n";
        return 0;
    }

    int vecNum = std::stoi(argv[1]);
    int vecDimm = std::stoi(argv[2]);

    int cpu_gpu = std::stoi(argv[3]);
    
    void (*fun)(int, int);

    if(cpu_gpu == 1){
        fun = &cpuMain;
    }else if(cpu_gpu==2){
        fun = &gpuMain;
    }else{
        std::cout<<"\nInvalid argument on 3rd place\n";
        return 0;
    }

    fun(vecNum, vecDimm);

    return 0;
}
