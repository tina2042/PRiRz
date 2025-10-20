#include "parallel_omp.hpp"
#include "sequential_proc.hpp"
#include <omp.h>
#include <algorithm> // Do std::min/max
#include <numeric>   // Do std::accumulate (chociaż nie używamy go tu)

// Funkcja pomocnicza z sequential_proc.cpp (potrzebna do CDF i transformacji)
// Pamiętaj, aby ją zaimplementować lub dołączyć w tym pliku!
// Dla uproszczenia zakładamy, że mamy dostęp do implementacji CDF i applyEqualization
std::vector<int> calculateCDF(const std::vector<int>& hist);
cv::Mat applyEqualization(const cv::Mat& inputImage, const std::vector<int>& cdf);


// ----------------------------------------------------------------------
// WERSJA W SKALI SZAROŚCI (GRYSCALE)
// ----------------------------------------------------------------------

std::vector<int> calculateHistogram_OMP_Grayscale(const cv::Mat& image) {
    if (image.channels() != 1) return {};

    const int NUM_BINS = 256;
    const int N_ROWS = image.rows;
    const int N_COLS = image.cols;
    
    // Pobierz maksymalną liczbę wątków
    int num_threads = omp_get_max_threads(); 
    
    // 1. Inicjalizacja Lokalnych Histogramów Wątków
    // Każdy wątek otrzymuje swój własny, zerowy wektor histogramu (256 elementów).
    // Używamy std::vector<std::vector<int>> do przechowywania lokalnych kopii.
    std::vector<std::vector<int>> local_hists(num_threads, std::vector<int>(NUM_BINS, 0));
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        std::vector<int>& my_hist = local_hists[tid];
        
        // 2. Równoległe Zliczanie
        // Podział pętli po wierszach na wątki (domyślnie dynamiczny/statyczny)
        #pragma omp for schedule(static)
        for (int i = 0; i < N_ROWS; ++i) {
            const uchar* rowPtr = image.ptr<uchar>(i);
            for (int j = 0; j < N_COLS; ++j) {
                my_hist[rowPtr[j]]++;
            }
        }
        
    } // Koniec bloku równoległego. W tym miejscu wątki się synchronizują.
    
    // 3. Redukcja (Sumowanie Lokalnych Histogramów do Globalnego)
    // Po zakończeniu pętli, wątek główny sumuje wyniki.
    std::vector<int> global_hist(NUM_BINS, 0);
    for (int t = 0; t < num_threads; ++t) {
        for (int b = 0; b < NUM_BINS; ++b) {
            global_hist[b] += local_hists[t][b];
        }
    }

    return global_hist;
}

cv::Mat equalize_OMP_Grayscale(const cv::Mat& inputImage) {
    // 1. Równoległe obliczenie histogramu
    std::vector<int> hist = calculateHistogram_OMP_Grayscale(inputImage);
    
    // 2. Sekwencyjne obliczenie CDF (jest szybkie, nie opłaca się paralelizować)
    std::vector<int> cdf = calculateCDF(hist);
    
    // 3. Sekwencyjne zastosowanie transformacji (można by paralelizować, ale dla spójności
    // i minimalizmu KM1 używamy sekwencyjnej wersji, która generuje LUT)
    return applyEqualization(inputImage, cdf);
}


// ----------------------------------------------------------------------
// WERSJA KOLOROWA (3 KANAŁY RGB)
// ----------------------------------------------------------------------

std::vector<std::vector<int>> calculateHistograms_OMP_Color(const cv::Mat& image) {
    if (image.channels() != 3) return {};
    
    const int NUM_BINS = 256;
    const int N_ROWS = image.rows;
    const int N_COLS = image.cols;
    
    int num_threads = omp_get_max_threads(); 

    // Lokalny histogram dla każdego kanału i każdego wątku: [wątek][kanał][poziom jasności]
    std::vector<std::vector<std::vector<int>>> local_hists(
        num_threads, std::vector<std::vector<int>>(3, std::vector<int>(NUM_BINS, 0))
    );
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        
        // Zliczanie pikseli
        #pragma omp for schedule(static)
        for (int i = 0; i < N_ROWS; ++i) {
            const uchar* rowPtr = image.ptr<uchar>(i);
            for (int j = 0; j < N_COLS; ++j) {
                // W OpenCV kanały są w kolejności BGR (0: B, 1: G, 2: R)
                local_hists[tid][0][rowPtr[j * 3 + 0]]++; // B
                local_hists[tid][1][rowPtr[j * 3 + 1]]++; // G
                local_hists[tid][2][rowPtr[j * 3 + 2]]++; // R
            }
        }
    }

    // Redukcja do Globalnego Histogramu (3 kanały)
    std::vector<std::vector<int>> global_hists(3, std::vector<int>(NUM_BINS, 0));
    for (int t = 0; t < num_threads; ++t) {
        for (int c = 0; c < 3; ++c) { // Iteracja po kanałach
            for (int b = 0; b < NUM_BINS; ++b) {
                global_hists[c][b] += local_hists[t][c][b];
            }
        }
    }
    
    return global_hists;
}

cv::Mat equalize_OMP_Color(const cv::Mat& inputImage) {
    // 1. Równoległe obliczenie histogramów dla 3 kanałów
    std::vector<std::vector<int>> hist_channels = calculateHistograms_OMP_Color(inputImage);
    
    cv::Mat equalizedImage = inputImage.clone();
    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels); // Rozdzielenie obrazu na 3 kanały BGR
    
    // 2. Obliczenie CDF i zastosowanie equalizacji dla każdego kanału osobno (sekwencyjnie)
    for (int c = 0; c < 3; ++c) {
        std::vector<int> cdf = calculateCDF(hist_channels[c]);
        // Używamy applyEqualization na pojedynczym kanale (Mat), co jest bezpieczne
        channels[c] = applyEqualization(channels[c], cdf); 
    }
    
    // 3. Złączenie kanałów z powrotem w jeden obraz kolorowy
    cv::merge(channels, equalizedImage);
    
    return equalizedImage;
}