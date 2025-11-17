#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>

#include "parallel_cuda.cuh"

// Funkcja pomocnicza na urządzeniu (device)
// Emuluje cv::saturate_cast<uchar> z zabezpieczeniem przed wartościami ujemnymi i > 255.
__device__ uchar __saturate_global_255(float x) {
    if (x < 0.0f) return 0;
    if (x > 255.0f) return 255;
    // Należy również dołączyć nagłówek <cmath> lub <math.h> dla roundf()
    return (uchar)roundf(x); 
}
// Kernel wykonujący sekwencyjną sumę prefiksową. 
// Jest to wydajne, ponieważ num_bins jest małe (max 256).
__global__ void prefixSumKernel(const int* hist_in, int* cdf_out, int num_bins) {
    // Zakładamy, że ten kernel jest uruchomiony na jednym wątku (gridDim.x = 1, blockDim.x = 1), 
    // ponieważ num_bins jest bardzo małe (max 256) i transformacja sekwencyjna jest natychmiastowa.

    // Pierwszy element CDF jest równy pierwszemu elementowi histogramu
    if (num_bins > 0) {
        cdf_out[0] = hist_in[0];
    }

    // Obliczanie sumy kumulatywnej
    for (int i = 1; i < num_bins; ++i) {
        cdf_out[i] = cdf_out[i - 1] + hist_in[i];
    }
}
// Kernel wykonujący sekwencyjną sumę prefiksową dla 3 kanałów
// Znowu, zakładamy, że num_bins jest małe (max 256), dlatego używamy sekwencyjnej transformacji
__global__ void prefixSumColorKernel(const int* histR_in, const int* histG_in, const int* histB_in,
                                     int* cdfR_out, int* cdfG_out, int* cdfB_out, int num_bins) {
    
    if (num_bins > 0) {
        // Inicjalizacja pierwszego elementu
        cdfR_out[0] = histR_in[0];
        cdfG_out[0] = histG_in[0];
        cdfB_out[0] = histB_in[0];
    }

    // Obliczanie sumy kumulatywnej dla wszystkich kanałów jednocześnie
    for (int i = 1; i < num_bins; ++i) {
        cdfR_out[i] = cdfR_out[i - 1] + histR_in[i];
        cdfG_out[i] = cdfG_out[i - 1] + histG_in[i];
        cdfB_out[i] = cdfB_out[i - 1] + histB_in[i];
    }
}

// Kernel obliczający tablicę LUT
__global__ void buildLUTKernel(const int* cdf, uchar* lut_out, int total_pixels, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAX_LEVELS = 256; // Rozmiar wyjściowego LUT

    if (tid >= MAX_LEVELS) return; // Wątek poza zakresem LUT

    // 1. Znajdź minimalną wartość CDF (cdf_min)
    // Ponieważ cdf jest posortowane, szukamy pierwszego elementu > 0.
    int cdf_min = 0;
    // Ze względu na mały rozmiar num_bins, możemy po prostu przeiterować cdf w jednym wątku
    // (choć efektywniejsze byłoby przekazanie cdf_min jako argumentu, 
    // lub użycie atomika/redukcji, jeśli cdf byłoby duże).
    // Dla prostoty, dla małego num_bins:
    for (int i = 0; i < num_bins; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // Zapobieganie dzieleniu przez zero
    if (total_pixels == cdf_min) { 
        lut_out[tid] = tid; // Nie zmieniaj nic, jeśli obraz jest jednolity
        return;
    }

    // 2. Skalowanie: Mapowanie oryginalnego piksela (tid, 0-255) do indeksu przedziału (bin_index, 0-num_bins-1)
    float scale_to_bin = (float)num_bins / (float)MAX_LEVELS;
    int bin_index = (int)(tid * scale_to_bin);
    
    // Zabezpieczenie przed przekroczeniem zakresu
    if (bin_index >= num_bins) bin_index = num_bins - 1;

    // 3. Wzór Equalizacji
    float cdf_value = (float)cdf[bin_index];
    
    // Nowa_Wartość = round( (CDF[i] - CDF_min) / (Total_Pixels - CDF_min) * 255 )
    float result = ((cdf_value - (float)cdf_min) / (total_pixels - (float)cdf_min)) * 255.0f;

    // 4. Zapis do LUT
    // cv::saturate_cast<uchar> to saturates_cast - prosta konwersja
    lut_out[tid] = (uchar)__saturate_global_255(result); 
}

// Kernel obliczający 3 tablice LUT (dla R, G, B) na raz
__global__ void buildLUTColorKernel(const int* cdfR, const int* cdfG, const int* cdfB,
                                    uchar* lutR_out, uchar* lutG_out, uchar* lutB_out, 
                                    int total_pixels, int num_bins) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAX_LEVELS = 256; 

    if (tid >= MAX_LEVELS) return; 

    // 1. Znajdź minimalne CDF (dla każdego kanału)
    int cdfR_min = 0, cdfG_min = 0, cdfB_min = 0;
    
    // Ponieważ num_bins jest małe, iterujemy (lub powinniśmy użyć pre-obliczonych wartości)
    for (int i = 0; i < num_bins; ++i) {
        if (cdfR[i] > 0 && cdfR_min == 0) cdfR_min = cdfR[i];
        if (cdfG[i] > 0 && cdfG_min == 0) cdfG_min = cdfG[i];
        if (cdfB[i] > 0 && cdfB_min == 0) cdfB_min = cdfB[i];
        // Można zoptymalizować: if (cdfR_min > 0 && cdfG_min > 0 && cdfB_min > 0) break;
    }
    
    // 2. Skalowanie
    float scale_to_bin = (float)num_bins / (float)MAX_LEVELS;
    int bin_index = (int)(tid * scale_to_bin);
    if (bin_index >= num_bins) bin_index = num_bins - 1;

    // 3. Wzór Equalizacji dla R, G, B
    float cdfR_val = (float)cdfR[bin_index];
    float cdfG_val = (float)cdfG[bin_index];
    float cdfB_val = (float)cdfB[bin_index];
    
    // LUT dla R
    if (total_pixels != cdfR_min) {
        float resultR = ((cdfR_val - cdfR_min) / (total_pixels - cdfR_min)) * 255.0f;
        lutR_out[tid] = __saturate_global_255(resultR); 
    } else { lutR_out[tid] = tid; }

    // LUT dla G
    if (total_pixels != cdfG_min) {
        float resultG = ((cdfG_val - cdfG_min) / (total_pixels - cdfG_min)) * 255.0f;
        lutG_out[tid] = __saturate_global_255(resultG); 
    } else { lutG_out[tid] = tid; }
    
    // LUT dla B
    if (total_pixels != cdfB_min) {
        float resultB = ((cdfB_val - cdfB_min) / (total_pixels - cdfB_min)) * 255.0f;
        lutB_out[tid] = __saturate_global_255(resultB); 
    } else { lutB_out[tid] = tid; }
}

// Kernel do zastosowania LUT (Nowy, brak go w oryginalnym kodzie!)
__global__ void applyLUTKernel(uchar* image_data, const uchar* lut, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        // Użyj oryginalnej wartości piksela jako indeksu do LUT
        // Zapisz nową wartość (z LUT) z powrotem do piksela
        image_data[i] = lut[image_data[i]];
    }
}

// Kernel do zastosowania 3 tablic LUT na obrazie BGR
__global__ void applyLUTColorKernel(uchar* image_data, const uchar* lutR, const uchar* lutG, const uchar* lutB, int size_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size_pixels; i += stride) {
        // Obraz jest w formacie BGR (jak w OpenCV)
        // B (indeks 3*i + 0)
        // G (indeks 3*i + 1)
        // R (indeks 3*i + 2)
        
        uchar valB = image_data[3 * i + 0];
        uchar valG = image_data[3 * i + 1];
        uchar valR = image_data[3 * i + 2];

        // Zastosowanie LUT
        image_data[3 * i + 0] = lutB[valB];
        image_data[3 * i + 1] = lutG[valG];
        image_data[3 * i + 2] = lutR[valR];
    }
}

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
    int total_pixels = width * height;
    const int MAX_LEVELS = 256;
    
    // Zabezpieczenie i obliczenie skali
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    // 1️⃣ Oblicz histogram na GPU (jak poprzednio, wymaga kopiowania do CPU)
    std::vector<int> histogramR(num_bins, 0), histogramG(num_bins, 0), histogramB(num_bins, 0);
    int numBlocks = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    
    // Wrapper, który przenosi obraz na GPU, liczy histogram i zwraca 3 histogramy do CPU
    computeHistogramCUDAColor(inputImage.data, histogramR.data(), histogramG.data(), histogramB.data(), 
                              width, height, threadsPerBlock, numBlocks, num_bins); 
    
    // 2️⃣ Alokacja pamięci GPU i kopiowanie
    int *d_histR, *d_histG, *d_histB;
    int *d_cdfR, *d_cdfG, *d_cdfB;
    uchar *d_lutR, *d_lutG, *d_lutB;
    uchar *d_image_data; // dla przetwarzania końcowego

    size_t hist_size_bytes = num_bins * sizeof(int);
    size_t lut_size_bytes = MAX_LEVELS * sizeof(uchar);

    cudaMalloc(&d_histR, hist_size_bytes);
    cudaMalloc(&d_histG, hist_size_bytes);
    cudaMalloc(&d_histB, hist_size_bytes);
    
    cudaMalloc(&d_cdfR, hist_size_bytes);
    cudaMalloc(&d_cdfG, hist_size_bytes);
    cudaMalloc(&d_cdfB, hist_size_bytes);

    cudaMalloc(&d_lutR, lut_size_bytes);
    cudaMalloc(&d_lutG, lut_size_bytes);
    cudaMalloc(&d_lutB, lut_size_bytes);
    
    cudaMalloc(&d_image_data, total_pixels * 3 * sizeof(uchar));
    
    // Kopiowanie 3 histogramów z CPU na GPU
    cudaMemcpy(d_histR, histogramR.data(), hist_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histG, histogramG.data(), hist_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histB, histogramB.data(), hist_size_bytes, cudaMemcpyHostToDevice);

    // --- CUDA: Obliczanie CDF (Prefix-Sum) ---
    // Kernel jest uruchomiony na jednym bloku/wątku.
    prefixSumColorKernel<<<1, 1>>>(d_histR, d_histG, d_histB, d_cdfR, d_cdfG, d_cdfB, num_bins);
    cudaDeviceSynchronize();

    // 3️⃣ Obliczanie 3 tablic LUT na GPU
    // Wywołujemy kernel dla każdej z 256 wartości w LUT (pojedynczy blok)
    buildLUTColorKernel<<<1, MAX_LEVELS>>>(d_cdfR, d_cdfG, d_cdfB, 
                                            d_lutR, d_lutG, d_lutB, 
                                            total_pixels, num_bins);
    cudaDeviceSynchronize();

    // 4️⃣ Zastosowanie LUT na GPU
    cv::Mat outputImage(height, width, CV_8UC3);
    
    // Kopiowanie oryginalnego obrazu (wektora) na GPU
    cudaMemcpy(d_image_data, inputImage.data, total_pixels * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
    
    // Zastosowanie transformacji:
    applyLUTColorKernel<<<numBlocks, threadsPerBlock>>>(d_image_data, d_lutR, d_lutG, d_lutB, total_pixels);
    cudaDeviceSynchronize();
    
    // Kopiowanie wynikowego obrazu z GPU do struktury OpenCV na CPU
    cudaMemcpy(outputImage.data, d_image_data, total_pixels * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci GPU
    cudaFree(d_histR); cudaFree(d_histG); cudaFree(d_histB);
    cudaFree(d_cdfR); cudaFree(d_cdfG); cudaFree(d_cdfB);
    cudaFree(d_lutR); cudaFree(d_lutG); cudaFree(d_lutB);
    cudaFree(d_image_data);

    return outputImage;
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
    int total_pixels = width * height;
    const int MAX_LEVELS = 256; // Rozmiar wyjściowego LUT

    // Zabezpieczenie num_bins
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));

    // 1️⃣ Oblicz histogram na GPU (jak poprzednio)
    std::vector<int> histogram(num_bins, 0);
    int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    
    // Używamy zaimplementowanego już wrappera
    computeHistogramCUDA(inputImage.data, histogram.data(), width, height, threadsPerBlock, numBlocks, num_bins);

    // 2️⃣ Kopiowanie histogramu na GPU i obliczenie CDF na GPU
    int* d_hist;
    int* d_cdf;
    uchar* d_lut;
    uchar* d_output_data;
    
    // Alokacja pamięci dla CDF i LUT (nowe!)
    cudaMalloc(&d_hist, num_bins * sizeof(int));
    cudaMalloc(&d_cdf, num_bins * sizeof(int));
    cudaMalloc(&d_lut, MAX_LEVELS * sizeof(uchar));
    
    // Kopiowanie histogramu z CPU na GPU
    cudaMemcpy(d_hist, histogram.data(), num_bins * sizeof(int), cudaMemcpyHostToDevice);

    // --- CUDA: Obliczanie CDF (Prefix-Sum) ---
    // Kernel jest uruchomiony na jednym bloku/wątku, bo num_bins jest małe
    prefixSumKernel<<<1, 1>>>(d_hist, d_cdf, num_bins);
    cudaDeviceSynchronize();

    // 3️⃣ Obliczanie LUT na GPU
    // Wywołujemy kernel dla każdej z 256 wartości w LUT
    buildLUTKernel<<<1, MAX_LEVELS>>>(d_cdf, d_lut, total_pixels, num_bins);
    cudaDeviceSynchronize();

    // 4️⃣ Zastosowanie LUT na GPU (nowy kernel)
    cv::Mat outputImage(height, width, CV_8UC1);
    
    // Alokacja miejsca na obraz na GPU
    cudaMalloc(&d_output_data, total_pixels * sizeof(uchar));
    
    // Kopiowanie oryginalnego obrazu (wektora) na GPU, aby go przetworzyć
    cudaMemcpy(d_output_data, inputImage.data, total_pixels * sizeof(uchar), cudaMemcpyHostToDevice);
    
    // Użyjemy kernele do zastosowania transformacji:
    applyLUTKernel<<<numBlocks, threadsPerBlock>>>(d_output_data, d_lut, total_pixels);
    cudaDeviceSynchronize();
    
    // Kopiowanie wynikowego obrazu z GPU do struktury OpenCV na CPU
    cudaMemcpy(outputImage.data, d_output_data, total_pixels * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Zwolnienie pamięci GPU
    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaFree(d_lut);
    cudaFree(d_output_data);

    return outputImage;
}