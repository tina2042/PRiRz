#include <iostream>
#include <fstream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <sys/stat.h> 
#include "parallel_mpi.hpp" 
#include "sequential_proc.hpp"

std::vector<int> calculateHistogram(const cv::Mat& image, int num_bins); 
std::vector<std::vector<int>> calculateColorHistogram(const cv::Mat& image, int num_bins); 

cv::Mat equalize_SEQ_Grayscale(const cv::Mat& inputImage, int num_bins);
cv::Mat equalize_SEQ_Color(const cv::Mat& inputImage, int num_bins);

void createDirectory(const std::string& path) {
    mkdir(path.c_str(), 0777); 
}

std::string generateUniqueFilename(const std::string& prefix, const std::string& outputDir) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_c);
    
    std::ostringstream filename_ss;
    filename_ss << prefix << "_" 
                << std::put_time(&tm, "%Y%m%d_%H%M%S") 
                << ".png";
                
    return outputDir + filename_ss.str();
}


int main(int argc, char** argv) {
    int DEFAULT_BINS = 256;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   if (argc < 3 || argc > 4) {
        if (rank == 0) {
            std::cerr << "Uzycie MPI: mpirun -np N ./mpi_runner <sciezka_do_obrazu> <tryb_MPI> [num_bins]" << std::endl;
            std::cerr << "Opcjonalny argument num_bins domyslnie: 256." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    if (argc == 4) {
        try {
            DEFAULT_BINS = std::stoi(argv[3]);
            if (DEFAULT_BINS <= 0 || DEFAULT_BINS > 256) {
                throw std::out_of_range("Liczba przedzialow poza zakresem (1-256).");
             }
         } catch (const std::invalid_argument& e) {
             if (rank == 0) std::cerr << "Blad: Argument num_bins nie jest poprawna liczba. Uzyto domyslnej wartosci: 256." << std::endl;
             DEFAULT_BINS = 256;
        } catch (const std::out_of_range& e) {
            if (rank == 0) std::cerr << "Blad: Liczba przedzialow poza zakresem. Uzyto domyslnej wartosci: 256." << std::endl;
         DEFAULT_BINS = 256;
        }
    }
    const std::string requested_mode = argv[2];
    
    cv::Mat outputImageMPI;
    double duration_mpi = 0;
    std::string filename_prefix = "MPI_";

    // ----------------------------------------------------------------------
    // 1. Ładowanie i Konfiguracja (Tylko Proces 0)
    // ----------------------------------------------------------------------
    cv::Mat inputImage;
    const std::string OUTPUT_DIR = "data/output/";

    if (rank == 0) {
        if (requested_mode == "MPI_COLOR") {
            inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
            filename_prefix += "COLOR";
            std::cout << "--- 7. Proces MPI (Color, " << size << " procesow) ---" << std::endl;
        } else {
            inputImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
            filename_prefix += "GRAY";
            std::cout << "--- 6. Proces MPI (Grayscale, " << size << " procesow) ---" << std::endl;
        }

        if (inputImage.empty()) {
            std::cerr << "Nie udalo sie zaladowac obrazu." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        createDirectory(OUTPUT_DIR);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ----------------------------------------------------------------------
    // 2. Wykonanie Logiki Równoległej
    // ----------------------------------------------------------------------
    if (requested_mode == "MPI_COLOR") {
        if (rank == 0) {
            double start_mpi = MPI_Wtime();
            outputImageMPI = equalize_MPI_Color(inputImage, rank, size, DEFAULT_BINS);
            double end_mpi = MPI_Wtime();
            duration_mpi = (end_mpi - start_mpi) * 1000.0;
        } else {
            equalize_MPI_Color(cv::Mat(), rank, size, DEFAULT_BINS);
        }

    } else { // Tryb MPI_GRAY
        if (rank == 0) {
            double start_mpi = MPI_Wtime();
            outputImageMPI = equalize_MPI_Grayscale(inputImage, rank, size, DEFAULT_BINS);
            double end_mpi = MPI_Wtime();
            duration_mpi = (end_mpi - start_mpi) * 1000.0;
        } else {
            equalize_MPI_Grayscale(cv::Mat(), rank, size, DEFAULT_BINS);
        }
    }
    
    // ----------------------------------------------------------------------
    // 3. Zapis Wyniku (Tylko Proces 0)
    // ----------------------------------------------------------------------
    if (rank == 0) {
        // --- 3A. Generowanie Wzorca Sekwencyjnego (dla weryfikacji) ---
        cv::Mat outputImageSEQ_Reference;
        std::string mode_label = "";
        
        if (requested_mode == "MPI_COLOR") {
            outputImageSEQ_Reference = equalize_SEQ_Color(inputImage, DEFAULT_BINS);
            mode_label = "Color";
        } else {
            outputImageSEQ_Reference = equalize_SEQ_Grayscale(inputImage, DEFAULT_BINS);
            mode_label = "Gray";
        }


        // --- 3B. Obliczanie Różnicy Histogramów ---
        long long total_diff = 0;

        if (mode_label == "Color") {
            // Weryfikacja dla kolorów (B+G+R)
            auto hist_mpi_color = calculateColorHistogram(outputImageMPI, DEFAULT_BINS);
            auto hist_seq_color = calculateColorHistogram(outputImageSEQ_Reference, DEFAULT_BINS);

            for (int c = 0; c < 3; ++c) {
                long long diff_channel = 0;
                for (int i = 0; i < 256; ++i)
                    diff_channel += std::abs(hist_seq_color[c][i] - hist_mpi_color[c][i]);
                total_diff += diff_channel;
                
                std::string channel_name = (c == 0 ? "B" : (c == 1 ? "G" : "R"));
                std::cout << "Różnica histogramów kanału " << channel_name 
                    << " SEQ vs MPI: " << diff_channel << std::endl; 
            }
            
            std::cout << "Różnica histogramów SEQ Color vs MPI Color: " << total_diff << std::endl;
        
        } else {
            auto hist_mpi_gray = calculateHistogram(outputImageMPI, DEFAULT_BINS);
            auto hist_seq_gray = calculateHistogram(outputImageSEQ_Reference, DEFAULT_BINS);

            for (int i = 0; i < 256; ++i)
                total_diff += std::abs(hist_seq_gray[i] - hist_mpi_gray[i]);
            
            std::cout << "Różnica histogramów SEQ vs MPI: " << total_diff << std::endl;
        }

        std::string filename_mpi = generateUniqueFilename(filename_prefix, OUTPUT_DIR);
        cv::imwrite(filename_mpi, outputImageMPI);
        
        std::cout << "Zapisano do: " << filename_mpi << std::endl; 
        std::cout << "Sredni czas wykonania (MPI): " << duration_mpi << " ms" << std::endl;       
    }
    
    MPI_Finalize();
    return 0;
}