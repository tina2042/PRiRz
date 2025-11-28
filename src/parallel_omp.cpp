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

std::vector<int> calculateHistogram_OMP_Grayscale(const cv::Mat& image, int num_bins) {
    if (image.channels() != 1) return {};

    // Ograniczamy num_bins do rozsądnego zakresu, np. 1 do 256
    num_bins = std::max(1, std::min(256, num_bins));
    const int N_ROWS = image.rows;
    const int N_COLS = image.cols;
    // Obliczanie współczynnika skalowania: mapuje 256 poziomów na num_bins indeksów
    const double scale = (double)num_bins / 256.0;
    
    // Pobierz maksymalną liczbę wątków
    int num_threads = omp_get_max_threads(); 
    
    // 1. Inicjalizacja Lokalnych Histogramów Wątków
    // Każdy wątek otrzymuje swój własny, zerowy wektor histogramu (256 elementów).
    // Używamy std::vector<std::vector<int>> do przechowywania lokalnych kopii.
    std::vector<std::vector<int>> local_hists(num_threads, std::vector<int>(num_bins, 0));
    
   #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        std::vector<int>& my_hist = local_hists[tid];
        
        // 2. Równoległe Zliczanie
        #pragma omp for schedule(static)
        for (int i = 0; i < N_ROWS; ++i) {
            const uchar* rowPtr = image.ptr<uchar>(i);
            for (int j = 0; j < N_COLS; ++j) {
                
                int pixel_value = rowPtr[j];
                
                // SKALOWANIE: Mapowanie 0-255 na 0-(num_bins-1)
                int bin_index = (int)(pixel_value * scale);
                
                // Zabezpieczenie przed przepełnieniem (choć matematycznie powinno być zbędne, 
                // jest to dobra praktyka ze względu na wartości 255 * scale)
                if (bin_index >= num_bins) bin_index = num_bins - 1;
                
                my_hist[bin_index]++;
            }
        }
        
    } // Koniec bloku równoległego.
    
    
    // 3. Redukcja (Sumowanie Lokalnych Histogramów do Globalnego)
    std::vector<int> global_hist(num_bins, 0);
    for (int t = 0; t < num_threads; ++t) {
        for (int b = 0; b < num_bins; ++b) { 
            global_hist[b] += local_hists[t][b];
        }
    }

    return global_hist;
}

cv::Mat equalize_OMP_Grayscale(const cv::Mat& inputImage, int num_bins) {
    const int MAX_LEVELS = 256;
    
    // Zabezpieczenie argumentu num_bins
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    // 1. Równoległe obliczenie histogramu
    std::vector<int> hist = calculateHistogram_OMP_Grayscale(inputImage, num_bins);
    
    // 2. Sekwencyjne obliczenie CDF (jest szybkie, nie opłaca się paralelizować)
    std::vector<int> cdf = calculateCDF(hist);
    
    // 3. Sekwencyjne zastosowanie transformacji (można by paralelizować, ale dla spójności
    // i minimalizmu KM1 używamy sekwencyjnej wersji, która generuje LUT)
    return applyEqualization(inputImage, cdf);
}


// ----------------------------------------------------------------------
// WERSJA KOLOROWA (3 KANAŁY RGB)
// ----------------------------------------------------------------------

std::vector<std::vector<int>> calculateHistograms_OMP_Color(const cv::Mat& image, int num_bins) {
    if (image.empty() || image.channels() != 3) return {};
    
    const int MAX_INTENSITY = 256;
    const int NUM_CHANNELS = 3;
    const int N_ROWS = image.rows;
    const int N_COLS = image.cols;
    
    // Ograniczamy num_bins
    num_bins = std::max(1, std::min(MAX_INTENSITY, num_bins));
    
    // Obliczanie współczynnika skalowania: mapuje 256 poziomów na num_bins indeksów
    const double scale = (double)num_bins / (double)MAX_INTENSITY;
    
    int num_threads = omp_get_max_threads(); 

    // 1. Inicjalizacja Lokalnych Histogramów Wątków
    // Histogramy mają rozmiar 'num_bins'
    std::vector<std::vector<std::vector<int>>> local_hists(
        num_threads, std::vector<std::vector<int>>(NUM_CHANNELS, std::vector<int>(num_bins, 0))
    );
    
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        
        // 2. Równoległe Zliczanie
        #pragma omp for schedule(static)
        for (int i = 0; i < N_ROWS; ++i) {
            const uchar* rowPtr = image.ptr<uchar>(i);
            for (int j = 0; j < N_COLS; ++j) {
                
                // Obliczanie indeksu dla każdego kanału
                int val_b = rowPtr[j * NUM_CHANNELS + 0];
                int val_g = rowPtr[j * NUM_CHANNELS + 1];
                int val_r = rowPtr[j * NUM_CHANNELS + 2];
                
                // SKALOWANIE: Mapowanie wartości piksela na indeks przedziału [0, num_bins-1]
                int bin_b = (int)(val_b * scale); 
                int bin_g = (int)(val_g * scale);
                int bin_r = (int)(val_r * scale);
                
                // Zabezpieczenie (choć rzadko potrzebne, jest bezpieczniejsze)
                if (bin_b >= num_bins) bin_b = num_bins - 1;
                if (bin_g >= num_bins) bin_g = num_bins - 1;
                if (bin_r >= num_bins) bin_r = num_bins - 1;

                // Zliczanie
                local_hists[tid][0][bin_b]++; // B
                local_hists[tid][1][bin_g]++; // G
                local_hists[tid][2][bin_r]++; // R
            }
        }
    }

    // 3. Redukcja do Globalnego Histogramu (3 kanały)
    // Globalny histogram ma również rozmiar 'num_bins'
    std::vector<std::vector<int>> global_hists(NUM_CHANNELS, std::vector<int>(num_bins, 0));
    
    for (int t = 0; t < num_threads; ++t) {
        for (int c = 0; c < NUM_CHANNELS; ++c) { // Iteracja po kanałach
            for (int b = 0; b < num_bins; ++b) { // Iteracja po num_bins
                global_hists[c][b] += local_hists[t][c][b];
            }
        }
    }
    
    return global_hists;
}

cv::Mat equalize_OMP_Color(const cv::Mat& inputImage, int num_bins) {
    const int MAX_LEVELS = 256;
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    // 1. Równoległe obliczenie histogramów dla 3 kanałów
    std::vector<std::vector<int>> hist_channels = calculateHistograms_OMP_Color(inputImage, num_bins);
    
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