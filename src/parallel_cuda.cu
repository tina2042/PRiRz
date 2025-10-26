#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include "parallel_cuda.cuh"

// Kernel CUDA - histogram grayscale
__global__ void histogramKernel(const unsigned char* image, int* hist, int size) {
    __shared__ int localHist[256];
    int tid = threadIdx.x;

    // Inicjalizacja lokalnego histogramu
    if (tid < 256) localHist[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Zliczanie pikseli
    for (int i = idx; i < size; i += stride) {
        atomicAdd(&localHist[image[i]], 1);
    }
    __syncthreads();

    // Redukcja do globalnego histogramu
    if (tid < 256) {
        atomicAdd(&hist[tid], localHist[tid]);
    }
}


// Wrapper C++ wywołujący kernel CUDA
extern "C" void computeHistogramCUDA(const unsigned char* input, int* histogram, int width, int height) {
    int size = width * height;
    
    unsigned char* d_image;
    int* d_hist;

    // Alokacja pamięci na GPU
    cudaMalloc(&d_image, size * sizeof(unsigned char));
    cudaMalloc(&d_hist, 256 * sizeof(int));

    // Zerowanie histogramu globalnego
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // Kopiowanie obrazu do GPU
    cudaMemcpy(d_image, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Uruchomienie kernela
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    histogramKernel<<<numBlocks, threadsPerBlock>>>(d_image, d_hist, size);
    cudaDeviceSynchronize();

    // Kopiowanie wyniku do CPU
    cudaMemcpy(histogram, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci GPU
    cudaFree(d_image);
    cudaFree(d_hist);
}

cv::Mat equalize_CUDA_Grayscale(const cv::Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    // 1️⃣ Oblicz histogram na GPU
    int histogram[256] = {0};
    computeHistogramCUDA(inputImage.data, histogram, width, height);

    // 2️⃣ Oblicz CDF na CPU
    std::vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i)
        cdf[i] = cdf[i - 1] + histogram[i];

    int total_pixels = width * height;
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    // 3️⃣ Utwórz LUT (look-up table)
    std::vector<unsigned char> lut(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = cv::saturate_cast<uchar>(
            ((float)(cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255.0f
        );
    }

    // 4️⃣ Utwórz wynikowy obraz (equalizowany)
    cv::Mat outputImage = inputImage.clone();
    for (int y = 0; y < height; ++y) {
        uchar* row = outputImage.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = lut[row[x]];
        }
    }

    return outputImage;
}