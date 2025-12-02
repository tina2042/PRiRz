#include "parallel_mpi.hpp"
#include <mpi.h>
#include "sequential_proc.hpp"

cv::Mat equalize_MPI_Grayscale(const cv::Mat& inputImage, int rank, int size, int num_bins) {
    
    const int MAX_INTENSITY = 256; 
    num_bins = std::max(1, std::min(MAX_INTENSITY, num_bins)); 
    
    const double scale = (double)num_bins / (double)MAX_INTENSITY;

    int total_rows = 0;
    int total_cols = 0;
    int total_pixels = 0;

    std::vector<int> sendcounts; 
    std::vector<int> displs;     

    int local_chunk_size = 0; 

    // ----------------------------------------------------------------------
    // 1. Dystrybucja Informacji o Rozmiarze i Obliczenie Dystrybucji
    // ----------------------------------------------------------------------
    if (rank == 0) {
        if (inputImage.empty() || inputImage.channels() != 1) return cv::Mat();

        total_rows = inputImage.rows;
        total_cols = inputImage.cols;
        total_pixels = total_rows * total_cols;

        sendcounts.resize(size);
        displs.resize(size);
        int current_displacement = 0;

        int base_rows_per_proc = total_rows / size;
        int remainder = total_rows % size;

        for (int i = 0; i < size; ++i) {
            int rows_for_proc = base_rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows_for_proc * total_cols; 
            displs[i] = current_displacement;
            current_displacement += sendcounts[i];
        }
    }
    
    // --- Rozesłanie danych konfiguracyjnych dla wszystkich procesów ---

    // 1a. Rozesłanie liczby kolumn (potrzebne do określenia rozmiaru wiersza)
    MPI_Bcast(&total_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 1b. Rozesłanie INDYWIDUALNEGO rozmiaru bloku (local_chunk_size) do każdego procesu 
    // poprzez wykorzystanie obliczonej wcześniej tablicy sendcounts i MPI_Scatter.
    // Używamy sendcounts z rank 0 jako danych wejściowych dla MPI_Scatter.
    // Na procesie 0: wysyłamy sendcounts[i] do procesu i.
    MPI_Scatter(
        rank == 0 ? sendcounts.data() : nullptr, 
        1,                                       
        MPI_INT, 
        &local_chunk_size,                       
        1, 
        MPI_INT, 
        0, 
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 2. Podział Danych i Alokacja Pamięci
    // ----------------------------------------------------------------------
    
    std::vector<uchar> local_data(local_chunk_size);
    
    uchar* global_data_ptr = (rank == 0) ? inputImage.data : nullptr;

    MPI_Scatterv(
        global_data_ptr,                           
        rank == 0 ? sendcounts.data() : nullptr,   
        rank == 0 ? displs.data() : nullptr,       
        MPI_UNSIGNED_CHAR,                         
        local_data.data(),                         
        local_chunk_size,                          
        MPI_UNSIGNED_CHAR,
        0,                                         
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 3. Obliczanie Częściowego Histogramu
    // ----------------------------------------------------------------------
    
    std::vector<int> local_hist(num_bins, 0); 
    
    for (int i = 0; i < local_chunk_size; ++i) { 
        int pixel_value = local_data[i];
        
        int bin_index = (int)(pixel_value * scale);
        
        if (bin_index >= num_bins) bin_index = num_bins - 1;
        
        local_hist[bin_index]++;
    }

    // ----------------------------------------------------------------------
    // 4. Sumowanie Histogramów (Redukcja)
    // ----------------------------------------------------------------------
    std::vector<int> global_hist(num_bins, 0); 

    MPI_Reduce(
        local_hist.data(), 
        global_hist.data(),
        num_bins, 
        MPI_INT, 
        MPI_SUM, 
        0, 
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 5. Finalne Przetwarzanie i Zwracanie Wyniku (Tylko proces 0)
    // ----------------------------------------------------------------------    
    if (rank == 0) {
        std::vector<int> cdf = calculateCDF(global_hist);
        cv::Mat outputImage = applyEqualization(inputImage, cdf); 
        return outputImage;
    }
    
    return cv::Mat();
}

cv::Mat equalize_MPI_Color(const cv::Mat& inputImage, int rank, int size, int num_bins) {
    
    const int MAX_INTENSITY = 256;
    const int NUM_CHANNELS = 3; 
    num_bins = std::max(1, std::min(MAX_INTENSITY, num_bins));
    const double scale = (double)num_bins / (double)MAX_INTENSITY;

    int total_pixels = 0;
    
    int chunk_size_pixels = 0;     
    int chunk_size_bytes = 0;      

    if (rank == 0) {
        if (inputImage.empty() || inputImage.channels() != NUM_CHANNELS) return cv::Mat();
        
        total_pixels = inputImage.rows * inputImage.cols;
        
        chunk_size_pixels = total_pixels / size;
        chunk_size_bytes = chunk_size_pixels * NUM_CHANNELS;
        
        if (total_pixels % size != 0) {
            std::cerr << "Blad: Liczba pikseli nie jest podzielna przez liczbe procesow." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return cv::Mat();
        }
    }

    MPI_Bcast(&chunk_size_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size_pixels = chunk_size_bytes / NUM_CHANNELS;
    
    // ----------------------------------------------------------------------
    // 1. Podział Danych (MPI_Scatter)
    // ----------------------------------------------------------------------
    std::vector<uchar> local_data(chunk_size_bytes);
    uchar* global_data_ptr = (rank == 0) ? inputImage.data : nullptr;

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
    std::vector<int> local_hists_flat(num_bins * NUM_CHANNELS, 0); 
    
    for (int i = 0; i < chunk_size_bytes; i += NUM_CHANNELS) {
        
        int bin_b = (int)(local_data[i + 0] * scale);
        int bin_g = (int)(local_data[i + 1] * scale);
        int bin_r = (int)(local_data[i + 2] * scale);

        if (bin_b >= num_bins) bin_b = num_bins - 1;
        if (bin_g >= num_bins) bin_g = num_bins - 1;
        if (bin_r >= num_bins) bin_r = num_bins - 1;

        local_hists_flat[0 * num_bins + bin_b]++; 
        local_hists_flat[1 * num_bins + bin_g]++; 
        local_hists_flat[2 * num_bins + bin_r]++; 
    }

    // ----------------------------------------------------------------------
    // 3. Sumowanie Histogramów (MPI_Reduce)
    // ----------------------------------------------------------------------

    std::vector<int> global_hists_flat(num_bins * NUM_CHANNELS, 0); 

    MPI_Reduce(
        local_hists_flat.data(), 
        global_hists_flat.data(),
        num_bins * NUM_CHANNELS, 
        MPI_INT, 
        MPI_SUM, 
        0, 
        MPI_COMM_WORLD
    );

    // ----------------------------------------------------------------------
    // 4. Finalne Przetwarzanie (Tylko proces 0)
    // ----------------------------------------------------------------------
    if (rank == 0) {
        std::vector<std::vector<int>> final_hists(NUM_CHANNELS);
        for (int c = 0; c < NUM_CHANNELS; ++c) {
            final_hists[c].assign(
                global_hists_flat.begin() + c * num_bins,
                global_hists_flat.begin() + (c + 1) * num_bins
            );
        }

        cv::Mat equalizedImage = inputImage.clone();
        std::vector<cv::Mat> channels;
        cv::split(inputImage, channels); 

        for (int c = 0; c < NUM_CHANNELS; ++c) {
            std::vector<int> cdf = calculateCDF(final_hists[c]);
            channels[c] = applyEqualization(channels[c], cdf); 
        }
        
        cv::merge(channels, equalizedImage);
        
        return equalizedImage;
    }
    
    return cv::Mat();
}