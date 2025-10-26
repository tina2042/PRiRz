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

// Kernel CUDA – histogram dla kolorowego obrazu
__global__ void histogramKernelColor(const uchar* image, int* histR, int* histG, int* histB, int size) {
    __shared__ int localR[256];
    __shared__ int localG[256];
    __shared__ int localB[256];

    int tid = threadIdx.x;

    // Inicjalizacja lokalnych histogramów
    if (tid < 256) {
        localR[tid] = 0;
        localG[tid] = 0;
        localB[tid] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        uchar r = image[3*i];
        uchar g = image[3*i + 1];
        uchar b = image[3*i + 2];

        atomicAdd(&localR[r], 1);
        atomicAdd(&localG[g], 1);
        atomicAdd(&localB[b], 1);
    }
    __syncthreads();

    if (tid < 256) {
        atomicAdd(&histR[tid], localR[tid]);
        atomicAdd(&histG[tid], localG[tid]);
        atomicAdd(&histB[tid], localB[tid]);
    }
}

// Wrapper C++ wywołujący kernel kolorowy
extern "C" void computeHistogramCUDAColor(const uchar* input, int* histR, int* histG, int* histB, int width, int height) {
    int size = width * height;

    uchar* d_image;
    int *d_histR, *d_histG, *d_histB;

    cudaMalloc(&d_image, size * 3 * sizeof(uchar));
    cudaMalloc(&d_histR, 256 * sizeof(int));
    cudaMalloc(&d_histG, 256 * sizeof(int));
    cudaMalloc(&d_histB, 256 * sizeof(int));

    cudaMemset(d_histR, 0, 256 * sizeof(int));
    cudaMemset(d_histG, 0, 256 * sizeof(int));
    cudaMemset(d_histB, 0, 256 * sizeof(int));

    cudaMemcpy(d_image, input, size * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    histogramKernelColor<<<numBlocks, threadsPerBlock>>>(d_image, d_histR, d_histG, d_histB, size);
    cudaDeviceSynchronize();

    cudaMemcpy(histR, d_histR, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histG, d_histG, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(histB, d_histB, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_histR);
    cudaFree(d_histG);
    cudaFree(d_histB);
}

// High-level CUDA equalization – kolor
cv::Mat equalize_CUDA_Color(const cv::Mat& inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    int histogramR[256] = {0}, histogramG[256] = {0}, histogramB[256] = {0};

    computeHistogramCUDAColor(inputImage.data, histogramR, histogramG, histogramB, width, height);

    // --- obliczenie CDF dla każdego kanału ---
    std::vector<int> cdfR(256), cdfG(256), cdfB(256);
    cdfR[0] = histogramR[0]; cdfG[0] = histogramG[0]; cdfB[0] = histogramB[0];

    for (int i = 1; i < 256; ++i) {
        cdfR[i] = cdfR[i-1] + histogramR[i];
        cdfG[i] = cdfG[i-1] + histogramG[i];
        cdfB[i] = cdfB[i-1] + histogramB[i];
    }

    int total_pixels = width * height;
    int cdfR_min = *std::find_if(cdfR.begin(), cdfR.end(), [](int x){ return x > 0; });
    int cdfG_min = *std::find_if(cdfG.begin(), cdfG.end(), [](int x){ return x > 0; });
    int cdfB_min = *std::find_if(cdfB.begin(), cdfB.end(), [](int x){ return x > 0; });

    std::vector<uchar> lutR(256), lutG(256), lutB(256);
    for (int i = 0; i < 256; ++i) {
        lutR[i] = cv::saturate_cast<uchar>(((float)(cdfR[i]-cdfR_min)/(total_pixels-cdfR_min))*255.0f);
        lutG[i] = cv::saturate_cast<uchar>(((float)(cdfG[i]-cdfG_min)/(total_pixels-cdfG_min))*255.0f);
        lutB[i] = cv::saturate_cast<uchar>(((float)(cdfB[i]-cdfB_min)/(total_pixels-cdfB_min))*255.0f);
    }

    cv::Mat output = inputImage.clone();
    for (int y = 0; y < height; ++y) {
        uchar* row = output.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            row[3*x]   = lutR[row[3*x]];
            row[3*x+1] = lutG[row[3*x+1]];
            row[3*x+2] = lutB[row[3*x+2]];
        }
    }

    return output;
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