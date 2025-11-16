// #include <iostream>
// #include <chrono>
// #include <ctime>
// #include <sstream>
// #include <iomanip> // Dodane dla std::put_time
// #include <sys/stat.h>
// #include <string>
// #include <functional>
// #include <omp.h>
// #include <fstream>
// #include <cuda_runtime.h>


// #include <opencv2/opencv.hpp>

// #include "sequential_proc.hpp" 
// #include "parallel_omp.hpp"
// #include "parallel_cuda.cuh"


// // Liczba powtórzeń dla uśrednienia
// const int NUM_RUNS = 10;

// /**
//  * @brief Funkcja pomocnicza mierząca średni czas wykonania danej operacji N razy.
//  * * @param op Funkcja lambda (lub inna wywoływalna), która wykonuje daną operację.
//  * @return long long Średni czas wykonania w milisekundach (ms).
//  */
// long long measureAverageTime(std::function<cv::Mat()> op) {
//     long long total_duration = 0;

//     // Wykonaj pomiar N razy
//     for (int i = 0; i < NUM_RUNS; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         // Wywołanie operacji (np. equalize_OMP_Grayscale)
//         op(); 
//         auto end = std::chrono::high_resolution_clock::now();
        
//         // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//         // total_duration += duration.count();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//         total_duration += duration;

//     }
    
//     // Zwróć średnią
//     return total_duration / NUM_RUNS;
// }

// // Funkcja pomocnicza do tworzenia katalogu, jeśli nie istnieje
// void createDirectory(const std::string& path) {
//     // 0777 to uprawnienia (odczyt, zapis, wykonanie dla wszystkich)
//     // S_IRWXU | S_IRWXG | S_IRWXO - bardziej przenośne uprawnienia, ale 0777 jest ok na WSL/Linux
//     mkdir(path.c_str(), 0777); 
// }

// // Funkcja pomocnicza do generowania unikalnej nazwy pliku
// std::string generateUniqueFilename(const std::string& prefix, const std::string& outputDir) {
//     auto now = std::chrono::system_clock::now();
//     std::time_t now_c = std::chrono::system_clock::to_time_t(now);
//     std::tm tm = *std::localtime(&now_c);
    
//     std::ostringstream filename_ss;
//     // Format: PREFIX_YYYYMMDD_HHMMSS.png
//     filename_ss << prefix << "_" 
//                 << std::put_time(&tm, "%Y%m%d_%H%M%S") 
//                 << ".png";
                
//     return outputDir + filename_ss.str();
// }


// /**
//  * @brief Zwraca listę bezpiecznych liczby wątków na blok dla obecnej karty CUDA.
//  *        Typowe wartości to wielokrotności 32 (warp size), nie większe niż maxThreadsPerBlock.
//  * @return std::vector<int> Lista możliwych threadsPerBlock
//  */
// std::vector<int> getSafeThreadCounts() {
//     int device;
//     cudaGetDevice(&device);

//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, device);

//     int maxThreads = prop.maxThreadsPerBlock;
//     int warpSize  = prop.warpSize; // zwykle 32

//     std::vector<int> safeCounts;

//     // Generujemy wielokrotności warpSize do maxThreads
//     for (int t = warpSize; t <= maxThreads; t += warpSize) {
//         safeCounts.push_back(t);
//     }

//     return safeCounts;
// }

// int main(int argc, char** argv) {
//     if (argc != 2) {
//         std::cerr << "Uzycie: " << argv[0] << " <sciezka_do_obrazu>" << std::endl;
//         return -1;
//     }

//     // --- Ustawienia Plików i Katalogów ---
//     const std::string OUTPUT_DIR = "data/output/";
//     createDirectory(OUTPUT_DIR);

//     // --- Ładowanie Obrazów (Raz, Minimalizując I/O) ---
//     // Ładujemy obraz raz w wersji szarej (do testów SEQ i OMP Grayscale)
//     cv::Mat inputImageGray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
//     if (inputImageGray.empty()) {
//         std::cerr << "Nie udalo sie zaladowac obrazu szarego: " << argv[1] << std::endl;
//         return -1;
//     }
//     // Ładujemy obraz raz w wersji kolorowej (do testów OMP Color)
//     cv::Mat inputImageColor = cv::imread(argv[1], cv::IMREAD_COLOR);
//     if (inputImageColor.empty()) {
//         std::cerr << "Nie udalo sie zaladowac obrazu kolorowego: " << argv[1] << std::endl;
//         return -1;
//     }


//     // ===================================================================
//     // 1. Wersja Sekwencyjna (Grayscale)
//     // ===================================================================
//     std::cout << "\n--- 1. Sekwencyjny Proces (Grayscale) ---" << std::endl;
    
//     cv::Mat outputImageSeq; // Wynik potrzebny do zapisu

//     // Mierzenie średniego czasu
//     long long duration_seq = measureAverageTime([&]() {
//         // Uwaga: Te funkcje nie zmieniają inputImageGray, więc są bezpieczne do powtarzania.
//         std::vector<int> hist_seq = calculateHistogram(inputImageGray);
//         std::vector<int> cdf_seq = calculateCDF(hist_seq);
//         outputImageSeq = applyEqualization(inputImageGray, cdf_seq);
//         return outputImageSeq; // Zwracamy wynik, by mieć co zapisać
//     });

//     // Zapis (tylko raz!)
//     std::string filename_seq = generateUniqueFilename("SEQ_GRAY", OUTPUT_DIR);
//     cv::imwrite(filename_seq, outputImageSeq);

//     std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
//             << duration_seq << " ms" << std::endl;
//     std::cout << "Zapisano do: " << filename_seq << std::endl;


//     // ===================================================================
// // 1b. Wersja Sekwencyjna (Color)
// // ===================================================================
// std::cout << "\n--- 1b. Sekwencyjny Proces (Color) ---" << std::endl;

// cv::Mat outputImageSeqColor;

// long long duration_seq_color = measureAverageTime([&]() {
//     // Oblicz histogramy kolorów
//     auto hist_seq_color = calculateColorHistogram(inputImageColor);

//     // Oblicz CDF dla każdego kanału
//     std::vector<std::vector<int>> cdf_seq_color(3);
//     for (int c = 0; c < 3; ++c)
//         cdf_seq_color[c] = calculateCDF(hist_seq_color[c]);

//     // Equalizacja dla każdego kanału
//     cv::Mat channels[3];
//     cv::split(inputImageColor, channels);
//     for (int c = 0; c < 3; ++c)
//         channels[c] = applyEqualization(channels[c], cdf_seq_color[c]);
//     cv::merge(channels, 3, outputImageSeqColor);

//     return outputImageSeqColor;
// });

// // Zapis
// std::string filename_seq_color = generateUniqueFilename("SEQ_COLOR", OUTPUT_DIR);
// cv::imwrite(filename_seq_color, outputImageSeqColor);

// std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): "
//           << duration_seq_color << " ms" << std::endl;
// std::cout << "Zapisano do: " << filename_seq_color << std::endl;



//     // ===================================================================
//     // 2. Wersja OpenMP (Grayscale)
//     // ===================================================================
//     std::cout << "\n--- 2. Proces OpenMP (Grayscale) ---" << std::endl;
    
//     cv::Mat outputImageOMPGray;

//     long long duration_omp_gray = measureAverageTime([&]() {
//         outputImageOMPGray = equalize_OMP_Grayscale(inputImageGray);
//         return outputImageOMPGray;
//     });

//     // Zapis
//     std::string filename_omp_gray = generateUniqueFilename("OMP_GRAY", OUTPUT_DIR);
//     cv::imwrite(filename_omp_gray, outputImageOMPGray);

//     std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
//             << duration_omp_gray << " ms" << std::endl;
//     std::cout << "Zapisano do: " << filename_omp_gray << std::endl;


//     std::cout << "\n--- Weryfikacja poprawności wyników OpenMP  - porównanie z wynikami sekwencyjnymi (dla grayscale) ---" << std::endl;
//     auto hist_seq = calculateHistogram(outputImageSeq);
//     auto hist_omp = calculateHistogram(outputImageOMPGray);

//     int diff = 0;
//     for (int i = 0; i < 256; ++i)
//         diff += std::abs(hist_seq[i] - hist_omp[i]);

//     std::cout << "Różnica histogramów: " << diff << std::endl;



//     // ===================================================================
//     // 3. Wersja OpenMP (Color)
//     // ===================================================================
//     std::cout << "\n--- 3. Proces OpenMP (Color) ---" << std::endl;

//     cv::Mat outputImageOMPColor;
    
//     long long duration_omp_color = measureAverageTime([&]() {
//         outputImageOMPColor = equalize_OMP_Color(inputImageColor);
//         return outputImageOMPColor;
//     });
    
//     // Zapis
//     std::string filename_omp_color = generateUniqueFilename("OMP_COLOR", OUTPUT_DIR);
//     cv::imwrite(filename_omp_color, outputImageOMPColor);
    
//     std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
//             << duration_omp_color << " ms" << std::endl;
//     std::cout << "Zapisano do: " << filename_omp_color << std::endl;


//     std::cout << "\n--- Sprawdzenie skalowalności OMP (Grayscale) ---" << std::endl;
//     std::ofstream results("scalability_results.csv");
//     results << "threads,time_ms\n";

   
//     int max_threads = omp_get_max_threads();

//     for (int threads = 1; threads <= max_threads; threads *= 2) {
//         omp_set_num_threads(threads);

//         long long duration = measureAverageTime([&]() {
//             equalize_OMP_Grayscale(inputImageGray);
//             return cv::Mat();
//         });

//         results << threads << "," << duration << "\n";
//         std::cout << "Wątki: " << threads << " -> " << duration << " ms" << std::endl;
//     }

//     results.close();

//     std::cout << "\n--- Sprawdzenie skalowalności OMP (Color) ---" << std::endl;
//     std::ofstream results_color("scalability_results_color.csv");
//     results_color << "threads,time_ms\n";

//     for (int threads = 1; threads <= max_threads; threads *= 2) {
//         omp_set_num_threads(threads);

//         long long duration = measureAverageTime([&]() {
//             equalize_OMP_Color(inputImageColor);
//             return cv::Mat();
//         });

//         results_color << threads << "," << duration << "\n";
//         std::cout << "Wątki: " << threads << " -> " << duration << " ms" << std::endl;
//     }

//     results_color.close();

   
//     // ===================================================================
//     // 4. Wersja CUDA (Grayscale)
//     // ===================================================================
//     std::cout << "\n--- 4. Proces CUDA (Grayscale) ---" << std::endl;

//     cv::Mat outputImageCUDA;

//     long long duration_cuda = measureAverageTime([&]() {
//         outputImageCUDA = equalize_CUDA_Grayscale(inputImageGray, 256);
//         return outputImageCUDA;
//     });

//     // Zapis
//     std::string filename_cuda = generateUniqueFilename("CUDA_GRAY", OUTPUT_DIR);
//     cv::imwrite(filename_cuda, outputImageCUDA);

//     std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runów): "
//             << duration_cuda << " ms" << std::endl;
//     std::cout << "Zapisano do: " << filename_cuda << std::endl;

//     // Weryfikacja poprawności z wersją sekwencyjną
//     std::cout << "\n--- Weryfikacja wyników CUDA ---" << std::endl;
//     auto hist_cuda = calculateHistogram(outputImageCUDA);

//     int diff_cuda = 0;
//     for (int i = 0; i < 256; ++i)
//         diff_cuda += std::abs(hist_seq[i] - hist_cuda[i]);

//     std::cout << "Różnica histogramów SEQ vs CUDA: " << diff_cuda << std::endl;


//     // ===================================================================
//     // 5. Wersja CUDA (Color)
//     // ===================================================================
//     std::cout << "\n--- 5. Proces CUDA (Color) ---" << std::endl;

//     cv::Mat outputImageCUDAColor;

//     long long duration_cuda_color = measureAverageTime([&]() {
//         outputImageCUDAColor = equalize_CUDA_Color(inputImageColor, 256);
//         return outputImageCUDAColor;
//     });

//     // Zapis
//     std::string filename_cuda_color = generateUniqueFilename("CUDA_COLOR", OUTPUT_DIR);
//     cv::imwrite(filename_cuda_color, outputImageCUDAColor);

//     std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runów): "
//             << duration_cuda_color << " ms" << std::endl;
//     std::cout << "Zapisano do: " << filename_cuda_color << std::endl;

//     // ===================================================================
//     // Weryfikacja wyników kolorowych (SEQ vs OMP Color)
//     // ===================================================================
//     std::cout << "\n--- Weryfikacja poprawności wyników (Color) ---" << std::endl;

//     // Oblicz histogramy dla SEQ Color
//     auto hist_seq_color = calculateColorHistogram(outputImageSeqColor);
//     // Oblicz histogramy dla OMP Color
//     auto hist_omp_color = calculateColorHistogram(outputImageOMPColor);

//     int total_diff_omp = 0;
//     // Porównujemy każdy kanał B, G, R
//     for (int c = 0; c < 3; ++c) {
//         int diff_channel = 0;
//         for (int i = 0; i < 256; ++i)
//             diff_channel += std::abs(hist_seq_color[c][i] - hist_omp_color[c][i]);
//         total_diff_omp += diff_channel;
//         std::string channel_name = (c == 0 ? "B" : (c == 1 ? "G" : "R"));
//         std::cout << "Różnica histogramów kanału " << channel_name 
//                 << " SEQ vs OMP: " << diff_channel << std::endl;
//     }

//     std::cout << std::endl;

//     std::cout << "Różnica histogramów (B+G+R) SEQ Color vs OMP Color: " 
//     << total_diff_omp << std::endl;
//     std::cout << std::endl;

//     // ===================================================================
//     // Weryfikacja wyników kolorowych (SEQ vs CUDA Color)
//     // ===================================================================
//     auto hist_cuda_color = calculateColorHistogram(outputImageCUDAColor);

//     int total_diff_cuda = 0;
//     for (int c = 0; c < 3; ++c) {
//         int diff_channel = 0;
//         for (int i = 0; i < 256; ++i)
//         diff_channel += std::abs(hist_seq_color[c][i] - hist_cuda_color[c][i]);
//     total_diff_cuda += diff_channel;
//     std::string channel_name = (c == 0 ? "B" : (c == 1 ? "G" : "R"));
//     std::cout << "Różnica histogramów kanału " << channel_name 
//     << " SEQ vs CUDA: " << diff_channel << std::endl;
//     }
//     std::cout << std::endl;

//     std::cout << "Różnica histogramów SEQ Color vs CUDA Color: " 
//             << total_diff_cuda << std::endl;



//     // ===================================================================
//     // Weryfikacja wyników kolorowych (OMP vs CUDA Color)
//     // ===================================================================
//     std::cout << "\n--- Weryfikacja wyników CUDA (Color) ---" << std::endl;

//     auto hist_cuda_R = calculateHistogram(outputImageCUDAColor); // dla uproszczenia używamy tej samej funkcji na wszystkich kanałach lub osobnej dla R/G/B
//     auto hist_omp_color1 = calculateHistogram(outputImageOMPColor);

//     int diff_cuda_color = 0;
//     for (int i = 0; i < 256; ++i)
//         diff_cuda_color += std::abs(hist_omp_color1[i] - hist_cuda_R[i]);

//     std::cout << "Różnica histogramów OMP Color vs CUDA Color: " << diff_cuda_color << std::endl;


//     std::cout << "\n--- Sprawdzenie skalowalności CUDA (Grayscale) ---" << std::endl;
//     std::ofstream results_cuda_gray("scalability_results_cuda_gray.csv");
//     results_cuda_gray << "threadsPerBlock,time_ms\n";

//     std::vector<int> threadOptions = {64, 128, 256, 512, 1024};

//     for (int threads : threadOptions) {
//         long long duration = measureAverageTime([&]() {
//             // Ustawienie liczby wątków w kernelu jest w środku computeHistogramCUDA
//             // Możemy dodać opcję przekazania threadsPerBlock do kernela, jeśli potrzebne
//             equalize_CUDA_Grayscale(inputImageGray, threads);
//             return cv::Mat();
//         });

//         results_cuda_gray << threads << "," << duration << "\n";
//         std::cout << "ThreadsPerBlock: " << threads << " -> " << duration << " ms" << std::endl;
//     }

//     results_cuda_gray.close();

//     std::cout << "\n--- Sprawdzenie skalowalności CUDA (Color) ---" << std::endl;
//     std::ofstream results_cuda_color("scalability_results_cuda_color.csv");
//     results_cuda_color << "threadsPerBlock,time_ms\n";

//     for (int threads : threadOptions) {
//         long long duration = measureAverageTime([&]() {
//             equalize_CUDA_Color(inputImageColor, threads);
//             return cv::Mat();
//         });

//         results_cuda_color << threads << "," << duration << "\n";
//         std::cout << "ThreadsPerBlock: " << threads << " -> " << duration << " ms" << std::endl;
//     }

//     results_cuda_color.close();


//     return 0;
// }




#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip> // Dodane dla std::put_time
#include <sys/stat.h>
#include <string>
#include <functional>
#include <omp.h>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "sequential_proc.hpp" 
#include "parallel_omp.hpp"
#include "parallel_cuda.cuh"


// Liczba powtórzeń dla uśrednienia
const int NUM_RUNS = 10;

/**
 * @brief Funkcja pomocnicza mierząca średni czas wykonania danej operacji N razy.
 * @param op Funkcja lambda (lub inna wywoływalna), która wykonuje daną operację.
 * @return long long Średni czas wykonania w milisekundach (ms).
 */
long long measureAverageTime(std::function<cv::Mat()> op) {
    long long total_duration = 0;

    // Wykonaj pomiar N razy
    for (int i = 0; i < NUM_RUNS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        // Wywołanie operacji (np. equalize_OMP_Grayscale)
        op(); 
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        total_duration += duration;
    }
    
    // Zwróć średnią
    return total_duration / NUM_RUNS;
}

// Funkcja pomocnicza do tworzenia katalogu, jeśli nie istnieje
void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0777); 
}

// Funkcja pomocnicza do generowania unikalnej nazwy pliku
std::string generateUniqueFilename(const std::string& prefix, const std::string& outputDir) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_c);
    
    std::ostringstream filename_ss;
    // Format: PREFIX_YYYYMMDD_HHMMSS.png
    filename_ss << prefix << "_" 
                << std::put_time(&tm, "%Y%m%d_%H%M%S") 
                << ".png";
                
    return outputDir + filename_ss.str();
}


/**
 * @brief Zwraca listę bezpiecznych liczby wątków na blok dla obecnej karty CUDA.
 *        Typowe wartości to wielokrotności 32 (warp size), nie większe niż maxThreadsPerBlock.
 * @return std::vector<int> Lista możliwych threadsPerBlock
 */
std::vector<int> getSafeThreadCounts() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxThreads = prop.maxThreadsPerBlock;
    int warpSize  = prop.warpSize; // zwykle 32

    std::vector<int> safeCounts;

    // Generujemy wielokrotności warpSize do maxThreads
    for (int t = warpSize; t <= maxThreads; t += warpSize) {
        safeCounts.push_back(t);
    }

    return safeCounts;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Uzycie: " << argv[0] << " <sciezka_do_obrazu> [tryb]" << std::endl;
        std::cerr << "Tryby: ALL, SEQ, OMP, CUDA, SEQ_OMP, SEQ_CUDA, OMP_CUDA, SCALING" << std::endl;
        return -1;
    }
    int DEFAULT_BINS = 256;
    if (argc == 4) {
        try {
            DEFAULT_BINS = std::stoi(argv[3]);
        } catch (...) {
            std::cerr << "Blad: Wystapil nieznany blad konwersji. Uzyto domyslnej wartosci: 256." << std::endl;
            DEFAULT_BINS = 256;
        }
    }
    std::string mode = "ALL";
    if (argc == 3) {
        mode = argv[2];
        // Normalize to uppercase for safety (very simple)
        std::transform(mode.begin(), mode.end(), mode.begin(), ::toupper);
    }

    // Helper: sprawdź czy wykonać daną sekcję
    auto shouldRun = [&](const std::initializer_list<std::string>& allowed)->bool {
        if (mode == "ALL") return true;
        for (auto &s : allowed) if (mode == s) return true;
        return false;
    };

    // --- Ustawienia Plików i Katalogów ---
    const std::string OUTPUT_DIR = "data/output/";
    createDirectory(OUTPUT_DIR);

    // --- Ładowanie Obrazów (Raz, Minimalizując I/O) ---
    cv::Mat inputImageGray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (inputImageGray.empty()) {
        std::cerr << "Nie udalo sie zaladowac obrazu szarego: " << argv[1] << std::endl;
        return -1;
    }
    cv::Mat inputImageColor = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImageColor.empty()) {
        std::cerr << "Nie udalo sie zaladowac obrazu kolorowego: " << argv[1] << std::endl;
        return -1;
    }

    // Flagi, żeby wiedzieć które bloki wykonały się (do bezpiecznych porównań)
    bool ran_seq = false;
    bool ran_seq_color = false;
    bool ran_omp_gray = false;
    bool ran_omp_color = false;
    bool ran_cuda_gray = false;
    bool ran_cuda_color = false;

    // Zmienne dla wyników (deklarujemy na poziomie main, użyjemy gdzie trzeba)
    cv::Mat outputImageSeq; 
    cv::Mat outputImageSeqColor;
    cv::Mat outputImageOMPGray;
    cv::Mat outputImageOMPColor;
    cv::Mat outputImageCUDA;
    cv::Mat outputImageCUDAColor;

    // ===================================================================
    // 1. Wersja Sekwencyjna (Grayscale)
    // ===================================================================
    if (shouldRun({"SEQ", "SEQ_OMP", "SEQ_CUDA"})) {
        std::cout << "\n--- 1. Sekwencyjny Proces (Grayscale) ---" << std::endl;

        long long duration_seq = measureAverageTime([&]() {
            std::vector<int> hist_seq = calculateHistogram(inputImageGray, DEFAULT_BINS);
            std::vector<int> cdf_seq = calculateCDF(hist_seq);
            outputImageSeq = applyEqualization(inputImageGray, cdf_seq);
            return outputImageSeq;
        });

        std::string filename_seq = generateUniqueFilename("SEQ_GRAY", OUTPUT_DIR);
        cv::imwrite(filename_seq, outputImageSeq);

        std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
                << duration_seq << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_seq << std::endl;

        ran_seq = true;
    }

    // ===================================================================
    // 1b. Wersja Sekwencyjna (Color)
    // ===================================================================
    if (shouldRun({"SEQ", "SEQ_OMP", "SEQ_CUDA"})) {
        std::cout << "\n--- 1b. Sekwencyjny Proces (Color) ---" << std::endl;

        long long duration_seq_color = measureAverageTime([&]() {
            auto hist_seq_color = calculateColorHistogram(inputImageColor, DEFAULT_BINS);

            std::vector<std::vector<int>> cdf_seq_color(3);
            for (int c = 0; c < 3; ++c)
                cdf_seq_color[c] = calculateCDF(hist_seq_color[c]);

            cv::Mat channels[3];
            cv::split(inputImageColor, channels);
            for (int c = 0; c < 3; ++c)
                channels[c] = applyEqualization(channels[c], cdf_seq_color[c]);
            cv::merge(channels, 3, outputImageSeqColor);

            return outputImageSeqColor;
        });

        std::string filename_seq_color = generateUniqueFilename("SEQ_COLOR", OUTPUT_DIR);
        cv::imwrite(filename_seq_color, outputImageSeqColor);

        std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): "
                  << duration_seq_color << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_seq_color << std::endl;

        ran_seq_color = true;
    }

    // ===================================================================
    // 2. Wersja OpenMP (Grayscale)
    // ===================================================================
    if (shouldRun({"OMP", "SEQ_OMP", "OMP_CUDA"})) {
        std::cout << "\n--- 2. Proces OpenMP (Grayscale) ---" << std::endl;

        long long duration_omp_gray = measureAverageTime([&]() {
            outputImageOMPGray = equalize_OMP_Grayscale(inputImageGray, DEFAULT_BINS);
            return outputImageOMPGray;
        });

        std::string filename_omp_gray = generateUniqueFilename("OMP_GRAY", OUTPUT_DIR);
        cv::imwrite(filename_omp_gray, outputImageOMPGray);

        std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
                << duration_omp_gray << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_omp_gray << std::endl;

        ran_omp_gray = true;

        // Weryfikacja poprawności wyników OpenMP  - porównanie z wynikami sekwencyjnymi (dla grayscale)
        if (ran_seq) {
            std::cout << "\n--- Weryfikacja poprawności wyników OpenMP  - porównanie z wynikami sekwencyjnymi (dla grayscale) ---" << std::endl;
            auto hist_seq_local = calculateHistogram(outputImageSeq, DEFAULT_BINS);
            auto hist_omp_local = calculateHistogram(outputImageOMPGray, DEFAULT_BINS);

            int diff = 0;
            for (int i = 0; i < 256; ++i)
                diff += std::abs(hist_seq_local[i] - hist_omp_local[i]);

            std::cout << "Różnica histogramów: " << diff << std::endl;
        } else {
            std::cout << "Brak wynikow SEQ do porownania z OMP (pomijam weryfikacje grayscale)." << std::endl;
        }
    }

    // ===================================================================
    // 3. Wersja OpenMP (Color)
    // ===================================================================
    if (shouldRun({"OMP", "SEQ_OMP", "OMP_CUDA"})) {
        std::cout << "\n--- 3. Proces OpenMP (Color) ---" << std::endl;

        long long duration_omp_color = measureAverageTime([&]() {
            outputImageOMPColor = equalize_OMP_Color(inputImageColor, DEFAULT_BINS);
            return outputImageOMPColor;
        });

        std::string filename_omp_color = generateUniqueFilename("OMP_COLOR", OUTPUT_DIR);
        cv::imwrite(filename_omp_color, outputImageOMPColor);

        std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
                << duration_omp_color << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_omp_color << std::endl;

        ran_omp_color = true;
    }

    // ===================================================================
    // 4. Wersja CUDA (Grayscale)
    // ===================================================================
    if (shouldRun({"CUDA", "SEQ_CUDA", "OMP_CUDA"})) {
        std::cout << "\n--- 4. Proces CUDA (Grayscale) ---" << std::endl;

        long long duration_cuda = measureAverageTime([&]() {
            outputImageCUDA = equalize_CUDA_Grayscale(inputImageGray, 256, DEFAULT_BINS);
            return outputImageCUDA;
        });

        std::string filename_cuda = generateUniqueFilename("CUDA_GRAY", OUTPUT_DIR);
        cv::imwrite(filename_cuda, outputImageCUDA);

        std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runów): "
                << duration_cuda << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_cuda << std::endl;

        ran_cuda_gray = true;

        // Weryfikacja poprawności z wersją sekwencyjną
        if (ran_seq) {
            std::cout << "\n--- Weryfikacja wyników CUDA (grayscale) ---" << std::endl;
            auto hist_cuda = calculateHistogram(outputImageCUDA, DEFAULT_BINS);
            auto hist_seq_local = calculateHistogram(outputImageSeq, DEFAULT_BINS);

            int diff_cuda = 0;
            for (int i = 0; i < 256; ++i)
                diff_cuda += std::abs(hist_seq_local[i] - hist_cuda[i]);

            std::cout << "Różnica histogramów SEQ vs CUDA: " << diff_cuda << std::endl;
        } else {
            std::cout << "Brak wynikow SEQ do porownania z CUDA (pomijam weryfikacje grayscale)." << std::endl;
        }
    }

    // ===================================================================
    // 5. Wersja CUDA (Color)
    // ===================================================================
    if (shouldRun({"CUDA", "SEQ_CUDA", "OMP_CUDA"})) {
        std::cout << "\n--- 5. Proces CUDA (Color) ---" << std::endl;

        long long duration_cuda_color = measureAverageTime([&]() {
            outputImageCUDAColor = equalize_CUDA_Color(inputImageColor, 256, DEFAULT_BINS);
            return outputImageCUDAColor;
        });

        std::string filename_cuda_color = generateUniqueFilename("CUDA_COLOR", OUTPUT_DIR);
        cv::imwrite(filename_cuda_color, outputImageCUDAColor);

        std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runów): "
                << duration_cuda_color << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_cuda_color << std::endl;

        ran_cuda_color = true;
    }

    // ===================================================================
    // Weryfikacja wyników kolorowych (SEQ vs OMP Color)
    // ===================================================================
    if (ran_seq_color && ran_omp_color) {
        std::cout << "\n--- Weryfikacja poprawności wyników (Color) ---" << std::endl;

        auto hist_seq_color = calculateColorHistogram(outputImageSeqColor, DEFAULT_BINS);
        auto hist_omp_color = calculateColorHistogram(outputImageOMPColor, DEFAULT_BINS);

        int total_diff_omp = 0;
        for (int c = 0; c < 3; ++c) {
            int diff_channel = 0;
            for (int i = 0; i < 256; ++i)
                diff_channel += std::abs(hist_seq_color[c][i] - hist_omp_color[c][i]);
            total_diff_omp += diff_channel;
            std::string channel_name = (c == 0 ? "B" : (c == 1 ? "G" : "R"));
            std::cout << "Różnica histogramów kanału " << channel_name 
                    << " SEQ vs OMP: " << diff_channel << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Różnica histogramów (B+G+R) SEQ Color vs OMP Color: " 
                  << total_diff_omp << std::endl;
        std::cout << std::endl;
    } else {
        if (!ran_seq_color && (ran_omp_color)) std::cout << "Brak SEQ_COLOR, pomijam porownania SEQ vs OMP (Color)." << std::endl;
    }

    // ===================================================================
    // Weryfikacja wyników kolorowych (SEQ vs CUDA Color)
    // ===================================================================
    if (ran_seq_color && ran_cuda_color) {
        auto hist_cuda_color = calculateColorHistogram(outputImageCUDAColor, DEFAULT_BINS);

        int total_diff_cuda = 0;
        for (int c = 0; c < 3; ++c) {
            int diff_channel = 0;
            for (int i = 0; i < 256; ++i)
                diff_channel += std::abs(calculateColorHistogram(outputImageSeqColor, DEFAULT_BINS)[c][i] - hist_cuda_color[c][i]);
            total_diff_cuda += diff_channel;
            std::string channel_name = (c == 0 ? "B" : (c == 1 ? "G" : "R"));
            std::cout << "Różnica histogramów kanału " << channel_name 
                      << " SEQ vs CUDA: " << diff_channel << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Różnica histogramów SEQ Color vs CUDA Color: " 
                  << total_diff_cuda << std::endl;
    } else {
        if (!ran_seq_color && ran_cuda_color) std::cout << "Brak SEQ_COLOR, pomijam porownania SEQ vs CUDA (Color)." << std::endl;
    }

    // ===================================================================
    // Weryfikacja wyników kolorowych (OMP vs CUDA Color)
    // ===================================================================
    if (ran_omp_color && ran_cuda_color) {
        std::cout << "\n--- Weryfikacja wyników CUDA (Color) ---" << std::endl;

        // Dla uproszczenia: porównujemy histogramy zliczone globalnie (nalezy dopracowac dla kanalow B/G/R)
        auto hist_cuda_R = calculateHistogram(outputImageCUDAColor, DEFAULT_BINS);
        auto hist_omp_color1 = calculateHistogram(outputImageOMPColor, DEFAULT_BINS);

        int diff_cuda_color = 0;
        for (int i = 0; i < 256; ++i)
            diff_cuda_color += std::abs(hist_omp_color1[i] - hist_cuda_R[i]);

        std::cout << "Różnica histogramów OMP Color vs CUDA Color: " << diff_cuda_color << std::endl;
    } else {
        if ((ran_omp_color && !ran_cuda_color) || (!ran_omp_color && ran_cuda_color))
            std::cout << "Brak jednego z wynikow OMP_COLOR/CUDA_COLOR, pomijam porownanie OMP vs CUDA (Color)." << std::endl;
    }

    // ===================================================================
    // Sprawdzenie skalowalności CUDA (Grayscale + Color) — tylko jeśli tryb to przewiduje (SCALING lub ALL)
    // Jeśli tryb był np. CUDA lub OMP_CUDA ale nie SCALING, to i tak powyższe CUDA/OMP testy się wykonały;
    // sekcje "skalowalności" uruchamiamy gdy użytkownik wybrał SCALING lub ALL lub też tryb zawiera OMP/CUDA (już powyżej uruchomione).
    // Zgodnie z zasadą A: skalowalności uruchamiamy gdy mode=="ALL" lub mode=="SCALING" lub gdy powyższe bloki działały i chcemy dodatkowo wykonać skalowalność.
    // W tej wersji: wykonujemy skalowalność jeśli mode == "ALL" lub mode == "SCALING" lub mode zawiera "OMP" -> OMP scaling; "CUDA" -> CUDA scaling.
    // ===================================================================
    // OMP scaling już wykonane wewnątrz bloku OMP (jeśli ten blok był wykonany) — ale upewnijmy się, że gdy użytkownik wybrał "SCALING" to wykonamy obie skalowalności.
    if (mode == "SCALING" || mode == "ALL") {
        // OMP skalowalność (jeśli OMP dostępne)
        std::cout << "\n--- Sprawdzenie skalowalności OMP (Grayscale) [tryb SCALING/ALL] ---" << std::endl;
        std::ofstream results_extra("scalability_results.csv");
        results_extra << "threads,time_ms\n";
        
        int max_threads = omp_get_max_threads();
        for (int threads = 1; threads <= max_threads; threads *= 2) {
            omp_set_num_threads(threads);
            
            long long duration = measureAverageTime([&]() {
                equalize_OMP_Grayscale(inputImageGray, DEFAULT_BINS);
                return cv::Mat();
            });
            
            results_extra << threads << "," << duration << "\n";
            std::cout << "Wątki: " << threads << " -> " << duration << " ms" << std::endl;
        }
        results_extra.close();
        
        std::cout << "\n--- Sprawdzenie skalowalności OMP (Color) [tryb SCALING/ALL] ---" << std::endl;
        std::ofstream results_extra_color("scalability_results_color.csv");
        results_extra_color << "threads,time_ms\n";
        for (int threads = 1; threads <= max_threads; threads *= 2) {
            omp_set_num_threads(threads);

            long long duration = measureAverageTime([&]() {
                equalize_OMP_Color(inputImageColor, DEFAULT_BINS);
                return cv::Mat();
            });

            results_extra_color << threads << "," << duration << "\n";
            std::cout << "Wątki: " << threads << " -> " << duration << " ms" << std::endl;
        }
        results_extra_color.close();

        // CUDA skalowalność
        std::cout << "\n--- Sprawdzenie skalowalności CUDA (Grayscale) [tryb SCALING/ALL] ---" << std::endl;
        std::ofstream results_cuda_gray("scalability_results_cuda_gray.csv");
        results_cuda_gray << "threadsPerBlock,time_ms\n";

        std::vector<int> threadOptions = {64, 128, 256, 512, 1024};
        for (int threads : threadOptions) {
            long long duration = measureAverageTime([&]() {
                equalize_CUDA_Grayscale(inputImageGray, threads, DEFAULT_BINS);
                return cv::Mat();
            });

            results_cuda_gray << threads << "," << duration << "\n";
            std::cout << "ThreadsPerBlock: " << threads << " -> " << duration << " ms" << std::endl;
        }
        results_cuda_gray.close();

        std::cout << "\n--- Sprawdzenie skalowalności CUDA (Color) [tryb SCALING/ALL] ---" << std::endl;
        std::ofstream results_cuda_color("scalability_results_cuda_color.csv");
        results_cuda_color << "threadsPerBlock,time_ms\n";

        for (int threads : threadOptions) {
            long long duration = measureAverageTime([&]() {
                equalize_CUDA_Color(inputImageColor, threads, DEFAULT_BINS);
                return cv::Mat();
            });

            results_cuda_color << threads << "," << duration << "\n";
            std::cout << "ThreadsPerBlock: " << threads << " -> " << duration << " ms" << std::endl;
        }
        results_cuda_color.close();
    }

    return 0;
}