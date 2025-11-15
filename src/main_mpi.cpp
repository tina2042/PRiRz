#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <sys/stat.h> 
#include "parallel_mpi.hpp" 
// Wymagane funkcje pomocnicze... (Zakładamy, że są zdefiniowane powyżej main lub w innym pliku)

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
    filename_ss << prefix << "_" 
                << std::put_time(&tm, "%Y%m%d_%H%M%S") 
                << ".png";
                
    return outputDir + filename_ss.str();
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Uzycie MPI: mpirun -np N ./mpi_runner <sciezka_do_obrazu> <tryb_MPI>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
    
    // Pobierz tryb raz
    const std::string requested_mode = argv[2];
    
    // Deklaracje zmiennych używanych do zapisu i mierzenia czasu
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
            outputImageMPI = equalize_MPI_Color(inputImage, rank, size);
            double end_mpi = MPI_Wtime();
            duration_mpi = (end_mpi - start_mpi) * 1000.0;
        } else {
            // Proces slave musi znać tryb
            equalize_MPI_Color(cv::Mat(), rank, size);
        }

    } else { // Tryb MPI_GRAY
        if (rank == 0) {
            double start_mpi = MPI_Wtime();
            outputImageMPI = equalize_MPI_Grayscale(inputImage, rank, size);
            double end_mpi = MPI_Wtime();
            duration_mpi = (end_mpi - start_mpi) * 1000.0;
        } else {
            equalize_MPI_Grayscale(cv::Mat(), rank, size);
        }
    }
    
    // ----------------------------------------------------------------------
    // 3. Zapis Wyniku (Tylko Proces 0)
    // ----------------------------------------------------------------------
    if (rank == 0) {
        std::string filename_mpi = generateUniqueFilename(filename_prefix, OUTPUT_DIR);
        cv::imwrite(filename_mpi, outputImageMPI);
        
        // Wypisanie wyniku w formacie oczekiwanym przez Pythona
        std::cout << "Sredni czas wykonania (MPI): " << duration_mpi << " ms" << std::endl;
        std::cout << "Zapisano do: " << filename_mpi << std::endl; 
    }
    
    MPI_Finalize();
    return 0;
}