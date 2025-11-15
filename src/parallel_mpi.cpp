#include "parallel_mpi.hpp"
#include <mpi.h>
#include "sequential_proc.hpp" // Dla calculateCDF i applyEqualization

// Definicje funkcji pomocniczych (zakładamy, że są w sequential_proc.cpp)
// std::vector<int> calculateCDF(const std::vector<int>& hist);
// cv::Mat applyEqualization(const cv::Mat& inputImage, const std::vector<int>& cdf);

cv::Mat equalize_MPI_Grayscale(const cv::Mat& inputImage, int rank, int size) {
    
    // Ustawienia dla 8-bitowego obrazu szarego
    const int NUM_BINS = 256; 
    
    // Zmienne używane tylko przez proces 0
    int total_rows = 0;
    int total_cols = 0;
    
    // Zmienne używane przez wszystkie procesy
    int rows_per_proc = 0;
    int chunk_size = 0; // Liczba pikseli, które każdy proces przetwarza
    
    // ----------------------------------------------------------------------
    // 1. Dystrybucja Informacji o Rozmiarze (tylko proces 0 wie o obrazie)
    // ----------------------------------------------------------------------
    if (rank == 0) {
        if (inputImage.empty() || inputImage.channels() != 1) return cv::Mat();
        
        total_rows = inputImage.rows;
        total_cols = inputImage.cols;
        
        // Obliczamy ile wierszy każdy proces powinien otrzymać
        rows_per_proc = total_rows / size;
        chunk_size = rows_per_proc * total_cols;
        
        // Sprawdzanie, czy wiersze są podzielne (uproszczenie)
        if (total_rows % size != 0) {
            std::cerr << "Blad: Liczba wierszy nie jest podzielna przez liczbe procesow. Wymagana jest modyfikacja dystrybucji." << std::endl;
            // W bardziej zaawansowanej wersji, proces 0 bierze resztę.
            // Dla projektu zakładamy podzielność dla uproszczenia dystrybucji.
            MPI_Abort(MPI_COMM_WORLD, 1);
            return cv::Mat();
        }
    }

    // Nadaj rozmiar bloku każdemu procesowi
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // ----------------------------------------------------------------------
    // 2. Podział Danych i Alokacja Pamięci
    // ----------------------------------------------------------------------
    
    // Zdefiniowanie bufora wejściowego dla procesu (dla jego części obrazu)
    std::vector<uchar> local_data(chunk_size);
    
    // Wskaźnik na cały obraz (tylko proces 0)
    uchar* global_data_ptr = nullptr;
    if (rank == 0) {
        global_data_ptr = inputImage.data;
    }

    // MPI_Scatter: Rozprosz obraz. 
    // Przesyłamy chunk_size pikseli z global_data_ptr do local_data.data()
    MPI_Scatter(
        global_data_ptr, // Bufor źródłowy (tylko na procesie 0)
        chunk_size,      // Liczba elementów wysyłana do każdego procesu
        MPI_UNSIGNED_CHAR, // Typ danych (uchar - 8-bitowy piksel)
        local_data.data(), // Bufor docelowy (na każdym procesie)
        chunk_size,      // Liczba elementów otrzymywana
        MPI_UNSIGNED_CHAR,
        0,               // Proces źródłowy (Root)
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 3. Obliczanie Częściowego Histogramu
    // ----------------------------------------------------------------------
    
    std::vector<int> local_hist(NUM_BINS, 0); // Lokalny histogram
    
    // Iteracja po swoim bloku danych
    for (int i = 0; i < chunk_size; ++i) {
        local_hist[local_data[i]]++;
    }

    // ----------------------------------------------------------------------
    // 4. Sumowanie Histogramów (Redukcja)
    // ----------------------------------------------------------------------

    std::vector<int> global_hist(NUM_BINS, 0); // Globalny histogram (tylko na procesie 0)

    // MPI_Reduce: Sumuj lokalne histogramy w globalny_hist na procesie 0
    MPI_Reduce(
        local_hist.data(), // Bufor źródłowy (local_hist)
        global_hist.data(),// Bufor docelowy (global_hist)
        NUM_BINS,          // Liczba elementów do zsumowania
        MPI_INT,           // Typ danych (liczniki są typu int)
        MPI_SUM,           // Operacja sumowania
        0,                 // Proces docelowy (Root)
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 5. Finalne Przetwarzanie i Zwracanie Wyniku (Tylko proces 0)
    // ----------------------------------------------------------------------
    
    if (rank == 0) {
        // Proces 0 kontynuuje, aby zakończyć equalizację:
        
        // 5a. Obliczenie CDF i LUT (sekwencyjne, ponieważ jest szybkie)
        std::vector<int> cdf = calculateCDF(global_hist);
        
        // 5b. Zastosowanie equalizacji na całym obrazie
        // Używamy sekwencyjnej funkcji applyEqualization z globalnym CDF
        cv::Mat outputImage = applyEqualization(inputImage, cdf); 
        
        return outputImage;
    }
    
    // Inne procesy zwracają pustą macierz, gdyż ich praca jest zakończona
    return cv::Mat();
}

cv::Mat equalize_MPI_Color(const cv::Mat& inputImage, int rank, int size) {
    
    const int NUM_BINS = 256; 
    const int NUM_CHANNELS = 3; 

    // Zmienne używane tylko przez proces 0
    int total_pixels = 0;
    
    // Zmienne używane przez wszystkie procesy
    int chunk_size_pixels = 0;     // Liczba pikseli, które każdy proces przetwarza
    int chunk_size_bytes = 0;      // Liczba bajtów (chunk_size_pixels * 3)

    if (rank == 0) {
        if (inputImage.empty() || inputImage.channels() != NUM_CHANNELS) return cv::Mat();
        
        total_pixels = inputImage.rows * inputImage.cols;
        
        // Obliczamy ile pikseli każdy proces powinien otrzymać
        chunk_size_pixels = total_pixels / size;
        chunk_size_bytes = chunk_size_pixels * NUM_CHANNELS;
        
        if (total_pixels % size != 0) {
            std::cerr << "Blad: Liczba pikseli nie jest podzielna przez liczbe procesow." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return cv::Mat();
        }
    }

    // Nadaj rozmiar bloku każdemu procesowi
    MPI_Bcast(&chunk_size_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size_pixels = chunk_size_bytes / NUM_CHANNELS;
    
    // ----------------------------------------------------------------------
    // 1. Podział Danych (MPI_Scatter)
    // ----------------------------------------------------------------------
    
    std::vector<uchar> local_data(chunk_size_bytes);
    uchar* global_data_ptr = (rank == 0) ? inputImage.data : nullptr;

    // Rozproszenie danych: chunk_size_bytes = 3 * chunk_size_pixels
    MPI_Scatter(
        global_data_ptr, 
        chunk_size_bytes, 
        MPI_UNSIGNED_CHAR, 
        local_data.data(), 
        chunk_size_bytes, 
        MPI_UNSIGNED_CHAR,
        0, 
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 2. Obliczanie Częściowych Histogramów (3 kanały)
    // ----------------------------------------------------------------------
    
    // Alokujemy jeden duży bufor do przechowywania trzech histogramów: [B | G | R]
    // Ten bufor ma rozmiar: 3 * 256
    std::vector<int> local_hists_flat(NUM_BINS * NUM_CHANNELS, 0); 
    
    // Zliczanie
    for (int i = 0; i < chunk_size_bytes; i += NUM_CHANNELS) {
        // W OpenCV format to BGR
        local_hists_flat[0 * NUM_BINS + local_data[i + 0]]++; // B
        local_hists_flat[1 * NUM_BINS + local_data[i + 1]]++; // G
        local_hists_flat[2 * NUM_BINS + local_data[i + 2]]++; // R
    }

    // ----------------------------------------------------------------------
    // 3. Sumowanie Histogramów (MPI_Reduce)
    // ----------------------------------------------------------------------

    // Globalny bufor do sumowania (tylko proces 0)
    std::vector<int> global_hists_flat(NUM_BINS * NUM_CHANNELS, 0); 

    // MPI_Reduce sumuje cały bufor (3 * 256 elementów) jednocześnie
    MPI_Reduce(
        local_hists_flat.data(), 
        global_hists_flat.data(),
        NUM_BINS * NUM_CHANNELS, // Redukujemy 768 elementów
        MPI_INT, 
        MPI_SUM, 
        0, 
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 4. Finalne Przetwarzanie (Tylko proces 0)
    // ----------------------------------------------------------------------
    
    if (rank == 0) {
        // 4a. Rozdzielenie płaskiego bufora na 3 wektory histogramów (dla CDF)
        std::vector<std::vector<int>> final_hists(NUM_CHANNELS);
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            final_hists[c].assign(
                global_hists_flat.begin() + c * NUM_BINS,
                global_hists_flat.begin() + (c + 1) * NUM_BINS
            );
        }

        // 4b. Podział obrazu i equalizacja każdego kanału
        cv::Mat equalizedImage = inputImage.clone();
        std::vector<cv::Mat> channels;
        cv::split(inputImage, channels); 

        for (int c = 0; c < NUM_CHANNELS; ++c) {
            std::vector<int> cdf = calculateCDF(final_hists[c]);
            channels[c] = applyEqualization(channels[c], cdf); 
        }
        
        // 4c. Złączenie kanałów
        cv::merge(channels, equalizedImage);
        
        return equalizedImage;
    }
    
    return cv::Mat();
}