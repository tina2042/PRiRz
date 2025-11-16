#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include "parallel_cuda.cuh"

// Kernel CUDA - histogram grayscale
__global__ void histogramKernel(const unsigned char* image, int* hist, int size, int num_bins, const float scale) {
    // Pamięć współdzielona alokowana dynamicznie
    extern __shared__ int localHist[]; // <--- Używamy dynamicznej alokacji
    int tid = threadIdx.x;

    // Inicjalizacja lokalnego histogramu
    if (tid < num_bins) localHist[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Zliczanie pikseli
    for (int i = idx; i < size; i += stride) {
        int pixel_value = image[i];
        
        // SKALOWANIE: Mapowanie 0-255 na 0-(num_bins-1)
        int bin_index = (int)(pixel_value * scale);
        if (bin_index >= num_bins) bin_index = num_bins - 1;
        
        atomicAdd(&localHist[bin_index], 1);
    }
    __syncthreads();

    // Redukcja do globalnego histogramu
    if (tid < num_bins) {
        atomicAdd(&hist[tid], localHist[tid]);
    }
}

// Kernel CUDA – histogram dla kolorowego obrazu
__global__ void histogramKernelColor(const uchar* image, int* histR, int* histG, int* histB, int size, int num_bins, const float scale) {
    extern __shared__ int localHist[];
    int *localB = localHist;
    int *localG = localHist + num_bins;
    int *localR = localHist + 2 * num_bins;

    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Inicjalizacja lokalnych histogramów
    // Inicjalizujemy wszystkie 3 * num_bins przedziałów
    for (int i = tid; i < num_bins * 3; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        // Obraz jest w formacie BGR (jak w OpenCV)
        uchar b = image[3*i + 0];
        uchar g = image[3*i + 1];
        uchar r = image[3*i + 2];

        // SKALOWANIE: Mapowanie 0-255 na 0-(num_bins-1)
        int bin_b = (int)(b * scale);
        int bin_g = (int)(g * scale);
        int bin_r = (int)(r * scale);
        
        // Zabezpieczenie (jest to minimalna wersja, w pełnej wersji najlepiej użyć min/max)
        if (bin_b >= num_bins) bin_b = num_bins - 1;
        if (bin_g >= num_bins) bin_g = num_bins - 1;
        if (bin_r >= num_bins) bin_r = num_bins - 1;

        atomicAdd(&localB[bin_b], 1);
        atomicAdd(&localG[bin_g], 1);
        atomicAdd(&localR[bin_r], 1);
    }
    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&histR[i], localR[i]);
        atomicAdd(&histG[i], localG[i]);
        atomicAdd(&histB[i], localB[i]);
    }
}

// Wrapper C++ wywołujący kernel kolorowy
extern "C" void computeHistogramCUDAColor(const uchar* input, int* histR, int* histG, int* histB, int width, int height, int threadsPerBlock, int numBlocks, int num_bins) {
    int size = width * height;
    float scale = (float)num_bins / 256.0f; // Obliczenie skali
    int hist_size_bytes = num_bins * sizeof(int);

    uchar* d_image;
    int *d_histR, *d_histG, *d_histB;

    cudaMalloc(&d_image, size * 3 * sizeof(uchar));
    cudaMalloc(&d_histR, hist_size_bytes); // <--- num_bins
    cudaMalloc(&d_histG, hist_size_bytes); // <--- num_bins
    cudaMalloc(&d_histB, hist_size_bytes); // <--- num_bins

    cudaMemset(d_histR, 0, hist_size_bytes);
    cudaMemset(d_histG, 0, hist_size_bytes);
    cudaMemset(d_histB, 0, hist_size_bytes);

    cudaMemcpy(d_image, input, size * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

    // Dynamiczna pamięć to 3 * num_bins * sizeof(int)
    int shared_mem_size = 3 * hist_size_bytes; 
    
    // Wywołanie kernela z dynamiczną pamięcią i skalą
    histogramKernelColor<<<numBlocks, threadsPerBlock, shared_mem_size>>>(
        d_image, d_histR, d_histG, d_histB, size, num_bins, scale
    );
    cudaDeviceSynchronize();

    cudaMemcpy(histR, d_histR, hist_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(histG, d_histG, hist_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(histB, d_histB, hist_size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_histR);
    cudaFree(d_histG);
    cudaFree(d_histB);
}

// High-level CUDA equalization – kolor
cv::Mat equalize_CUDA_Color(const cv::Mat& inputImage, int threadsPerBlock, int num_bins) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    const int MAX_LEVELS = 256;
    
    // Zabezpieczenie i obliczenie skali
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    const float scale_to_bin = (float)num_bins / (float)MAX_LEVELS; // 256 -> num_bins
    
    // Ustalenie LUT Size (LUT zawsze odwzorowuje PIKSEL->NOWA_WARTOŚĆ, musi być 256)
    // Zmienimy rozmiar tymczasowych LUT w kroku 3
    
    // 1️⃣ Oblicz histogram na GPU
    std::vector<int> histogramR(num_bins, 0), histogramG(num_bins, 0), histogramB(num_bins, 0);

    int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    // Wywołanie wrappera z num_bins (zakładamy, że wrapper i kernel są już poprawne)
    computeHistogramCUDAColor(inputImage.data, histogramR.data(), histogramG.data(), histogramB.data(), 
                              width, height, threadsPerBlock, numBlocks, num_bins); 
    
    // 2️⃣ Oblicz CDF na CPU (rozmiar num_bins)
    std::vector<int> cdfR(num_bins), cdfG(num_bins), cdfB(num_bins);
    cdfR[0] = histogramR[0]; cdfG[0] = histogramG[0]; cdfB[0] = histogramB[0];

    for (int i = 1; i < num_bins; ++i) { // Pętla idzie do num_bins
        cdfR[i] = cdfR[i-1] + histogramR[i];
        cdfG[i] = cdfG[i-1] + histogramG[i];
        cdfB[i] = cdfB[i-1] + histogramB[i];
    }

    int total_pixels = width * height;
    // ... (obliczenie cdf_min, które nadal używa num_bins) ...
    // (Używanie std::find_if na cdf o rozmiarze num_bins jest poprawne)
    int cdfR_min = *std::find_if(cdfR.begin(), cdfR.end(), [](int x){ return x > 0; });
    int cdfG_min = *std::find_if(cdfG.begin(), cdfG.end(), [](int x){ return x > 0; });
    int cdfB_min = *std::find_if(cdfB.begin(), cdfB.end(), [](int x){ return x > 0; });

    // 3️⃣ Utwórz LUT (Look-Up Table) o rozmiarze 256
    // LUT MUSI mieć rozmiar 256, aby móc być indeksowana przez oryginalny piksel 0-255.
    std::vector<uchar> lutR(MAX_LEVELS), lutG(MAX_LEVELS), lutB(MAX_LEVELS);
    
    // Wzór equalizacji używa CDF o rozmiarze num_bins. 
    // Musimy skalować LUT z 0...num_bins-1 do 0...255.
    const float scale_to_level = (float)MAX_LEVELS / (float)num_bins; // num_bins -> 256

    for (int i = 0; i < MAX_LEVELS; ++i) { 
        // Oblicz indeks przedziału bin, do którego należy wartość i (0-255)
        int bin_index = (int)(i * scale_to_bin); 
        
        // Zabezpieczenie
        if (bin_index >= num_bins) bin_index = num_bins - 1;

        // Używamy CDF z obliczonego bin_index, a nie i
        lutR[i] = cv::saturate_cast<uchar>(((float)(cdfR[bin_index]-cdfR_min)/(total_pixels-cdfR_min))*255.0f);
        lutG[i] = cv::saturate_cast<uchar>(((float)(cdfG[bin_index]-cdfG_min)/(total_pixels-cdfG_min))*255.0f);
        lutB[i] = cv::saturate_cast<uchar>(((float)(cdfB[bin_index]-cdfB_min)/(total_pixels-cdfB_min))*255.0f);
    }

    // 4️⃣ Zastosowanie LUT
    cv::Mat output = inputImage.clone();
    for (int y = 0; y < height; ++y) {
        uchar* row = output.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            // POPRAWKA 2: Teraz możemy używać oryginalnej wartości piksela jako indeksu
            // Ponieważ lutR ma rozmiar 256!
            row[3*x]= lutR[row[3*x]];
            row[3*x+1] = lutG[row[3*x+1]];
            row[3*x+2] = lutB[row[3*x+2]];
        }
    }

    return output;
}


// Wrapper C++ wywołujący kernel CUDA
extern "C" void computeHistogramCUDA(const unsigned char* input, int* histogram, int width, int height, int threadsPerBlock, int numBlocks, int num_bins) {
    int size = width * height;
    float scale = (float)num_bins / 256.0f; // Obliczenie skali
    
    unsigned char* d_image;
    int* d_hist;

    // Alokacja pamięci na GPU
    cudaMalloc(&d_image, size * sizeof(unsigned char));
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    // Zerowanie histogramu globalnego
    cudaMemset(d_hist, 0, num_bins * sizeof(int));

    // Kopiowanie obrazu do GPU
    cudaMemcpy(d_image, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Wywołanie kernela z dynamiczną pamięcią i skalą
    histogramKernel<<<numBlocks, threadsPerBlock, num_bins * sizeof(int)>>>(d_image, d_hist, size, num_bins, scale); 
    cudaDeviceSynchronize();

    // Kopiowanie wyniku do CPU
    cudaMemcpy(histogram, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci GPU
    cudaFree(d_image);
    cudaFree(d_hist);
}

cv::Mat equalize_CUDA_Grayscale(const cv::Mat& inputImage, int threadsPerBlock, int num_bins) {
    int width = inputImage.cols;
    int height = inputImage.rows;

    // 1️⃣ Oblicz histogram na GPU
    int histogram[num_bins];
    std::fill(histogram, histogram + num_bins, 0);
    int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    computeHistogramCUDA(inputImage.data, histogram, width, height, threadsPerBlock, numBlocks, num_bins);

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