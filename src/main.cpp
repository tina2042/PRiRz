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


#include <opencv2/opencv.hpp>

#include "sequential_proc.hpp" 
#include "parallel_omp.hpp"

// Liczba powtórzeń dla uśrednienia
const int NUM_RUNS = 10;

/**
 * @brief Funkcja pomocnicza mierząca średni czas wykonania danej operacji N razy.
 * * @param op Funkcja lambda (lub inna wywoływalna), która wykonuje daną operację.
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
        
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // total_duration += duration.count();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        total_duration += duration;

    }
    
    // Zwróć średnią
    return total_duration / NUM_RUNS;
}

// Funkcja pomocnicza do tworzenia katalogu, jeśli nie istnieje
void createDirectory(const std::string& path) {
    // 0777 to uprawnienia (odczyt, zapis, wykonanie dla wszystkich)
    // S_IRWXU | S_IRWXG | S_IRWXO - bardziej przenośne uprawnienia, ale 0777 jest ok na WSL/Linux
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

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uzycie: " << argv[0] << " <sciezka_do_obrazu>" << std::endl;
        return -1;
    }

    // --- Ustawienia Plików i Katalogów ---
    const std::string OUTPUT_DIR = "data/output/";
    createDirectory(OUTPUT_DIR);

    // --- Ładowanie Obrazów (Raz, Minimalizując I/O) ---
    // Ładujemy obraz raz w wersji szarej (do testów SEQ i OMP Grayscale)
    cv::Mat inputImageGray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (inputImageGray.empty()) {
        std::cerr << "Nie udalo sie zaladowac obrazu szarego: " << argv[1] << std::endl;
        return -1;
    }
    // Ładujemy obraz raz w wersji kolorowej (do testów OMP Color)
    cv::Mat inputImageColor = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (inputImageColor.empty()) {
        std::cerr << "Nie udalo sie zaladowac obrazu kolorowego: " << argv[1] << std::endl;
        return -1;
    }


    // ===================================================================
    // 1. Wersja Sekwencyjna (Grayscale)
    // ===================================================================
    std::cout << "\n--- 1. Sekwencyjny Proces (Grayscale) ---" << std::endl;
    
    cv::Mat outputImageSeq; // Wynik potrzebny do zapisu

    // Mierzenie średniego czasu
    long long duration_seq = measureAverageTime([&]() {
        // Uwaga: Te funkcje nie zmieniają inputImageGray, więc są bezpieczne do powtarzania.
        std::vector<int> hist_seq = calculateHistogram(inputImageGray);
        std::vector<int> cdf_seq = calculateCDF(hist_seq);
        outputImageSeq = applyEqualization(inputImageGray, cdf_seq);
        return outputImageSeq; // Zwracamy wynik, by mieć co zapisać
    });

    // Zapis (tylko raz!)
    std::string filename_seq = generateUniqueFilename("SEQ_GRAY", OUTPUT_DIR);
    cv::imwrite(filename_seq, outputImageSeq);

    std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
            << duration_seq << " ms" << std::endl;
    std::cout << "Zapisano do: " << filename_seq << std::endl;


    // ===================================================================
    // 2. Wersja OpenMP (Grayscale)
    // ===================================================================
    std::cout << "\n--- 2. Proces OpenMP (Grayscale) ---" << std::endl;
    
    cv::Mat outputImageOMPGray;

    long long duration_omp_gray = measureAverageTime([&]() {
        outputImageOMPGray = equalize_OMP_Grayscale(inputImageGray);
        return outputImageOMPGray;
    });

    // Zapis
    std::string filename_omp_gray = generateUniqueFilename("OMP_GRAY", OUTPUT_DIR);
    cv::imwrite(filename_omp_gray, outputImageOMPGray);

    std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
            << duration_omp_gray << " ms" << std::endl;
    std::cout << "Zapisano do: " << filename_omp_gray << std::endl;


    std::cout << "\n--- Weryfikacja poprawności wyników OpenMP  - porównanie z wynikami sekwencyjnymi (dla grayscale) ---" << std::endl;
    auto hist_seq = calculateHistogram(outputImageSeq);
    auto hist_omp = calculateHistogram(outputImageOMPGray);

    int diff = 0;
    for (int i = 0; i < 256; ++i)
        diff += std::abs(hist_seq[i] - hist_omp[i]);

    std::cout << "Różnica histogramów: " << diff << std::endl;



    // ===================================================================
    // 3. Wersja OpenMP (Color)
    // ===================================================================
    std::cout << "\n--- 3. Proces OpenMP (Color) ---" << std::endl;

    cv::Mat outputImageOMPColor;
    
    long long duration_omp_color = measureAverageTime([&]() {
        outputImageOMPColor = equalize_OMP_Color(inputImageColor);
        return outputImageOMPColor;
    });
    
    // Zapis
    std::string filename_omp_color = generateUniqueFilename("OMP_COLOR", OUTPUT_DIR);
    cv::imwrite(filename_omp_color, outputImageOMPColor);
    
    std::cout << "Sredni czas wykonania (" << NUM_RUNS << " runow): " 
            << duration_omp_color << " ms" << std::endl;
    std::cout << "Zapisano do: " << filename_omp_color << std::endl;


    std::cout << "\n--- Sprawdzenie skalowalności (Grayscale) ---" << std::endl;
    std::ofstream results("scalability_results.csv");
    results << "threads,time_ms\n";

   
    int max_threads = omp_get_max_threads();

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        omp_set_num_threads(threads);

        long long duration = measureAverageTime([&]() {
            equalize_OMP_Grayscale(inputImageGray);
            return cv::Mat();
        });

        results << threads << "," << duration << "\n";
        std::cout << "Wątki: " << threads << " -> " << duration << " ms" << std::endl;
    }

    results.close();

    std::cout << "\n--- Sprawdzenie skalowalności (Color) ---" << std::endl;
    std::ofstream results_color("scalability_results_color.csv");
    results_color << "threads,time_ms\n";

    for (int threads = 1; threads <= max_threads; threads *= 2) {
        omp_set_num_threads(threads);

        long long duration = measureAverageTime([&]() {
            equalize_OMP_Color(inputImageColor);
            return cv::Mat();
        });

        results_color << threads << "," << duration << "\n";
        std::cout << "Wątki: " << threads << " -> " << duration << " ms" << std::endl;
    }

    results_color.close();

   

    return 0;
}