#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <sys/stat.h>
#include "sequential_proc.hpp" 
// (zakładamy, że funkcje zostały zadeklarowane w sequential_proc.hpp)

// Funkcja pomocnicza do tworzenia katalogu, jeśli nie istnieje
void createDirectory(const std::string& path) {
    // 0777 to uprawnienia (odczyt, zapis, wykonanie dla wszystkich)
    mkdir(path.c_str(), 0777); 
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uzycie: " << argv[0] << " <sciezka_do_obrazu>" << std::endl;
        return -1;
    }

    // --- Ustawienia Plików i Katalogów ---
    const std::string OUTPUT_DIR = "data/output/";
    createDirectory(OUTPUT_DIR);

    // Generowanie unikalnej nazwy pliku na podstawie czasu
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_c);
    
    std::ostringstream filename_ss;
    // Format: SEQ_YYYYMMDD_HHMMSS.png
    filename_ss << "SEQ_" 
                << std::put_time(&tm, "%Y%m%d_%H%M%S") 
                << ".png";
                
    std::string output_filename = OUTPUT_DIR + filename_ss.str();
    
    // --- Ładowanie Obrazu ---
    cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Nie udalo sie zaladowac obrazu: " << argv[1] << std::endl;
        return -1;
    }

    // --- Sekwencyjny Proces Equalizacji ---
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> hist = calculateHistogram(inputImage);
    std::vector<int> cdf = calculateCDF(hist);
    cv::Mat outputImage = applyEqualization(inputImage, cdf);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // --- Zapis i Wyniki ---
    cv::imwrite(output_filename, outputImage);
    
    std::cout << "--- Wyniki Sekwencyjne ---" << std::endl;
    std::cout << "Czas wykonania: " << duration.count() << " ms" << std::endl;
    std::cout << "Zapisano obraz do: " << output_filename << std::endl;
    
    return 0;
}