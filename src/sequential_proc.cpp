#include <opencv2/opencv.hpp>
#include <vector>

std::vector<int> calculateHistogram(const cv::Mat& image, int num_bins) {
    // Histogram 256 przedziałów (dla 8-bitowego obrazu)
    std::vector<int> hist(num_bins, 0);
    double scale = (double)num_bins / 256.0;
    
    // Obraz powinien być w skali szarości (CV_8UC1)
    if (image.channels() != 1) {
        // Obsługa błędu lub konwersja
        return hist; 
    }
    
    // Iteracja po pikselach obrazu
    for (int i = 0; i < image.rows; ++i) {
        const uchar* rowPtr = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; ++j) {
            int pixelValue = rowPtr[j];
            int bin_index = (int)(pixelValue * scale); // Skalowanie
            
            // Upewnienie się, że indeks jest poprawny
            if (bin_index >= num_bins) bin_index = num_bins - 1; 
            
            hist[bin_index]++;
        }
    }
    
    return hist;
}


// Zliczanie histogramu dla obrazów kolorowych (BGR)
std::vector<std::vector<int>> calculateColorHistogram(const cv::Mat& image, int num_bins) {
    // Sprawdzenie, czy num_bins jest rozsądne (przynajmniej 1, max 256)
    if (num_bins < 1 || num_bins > 256) {
        num_bins = 256; // Użyj wartości domyślnej
    }
    // Alokacja: 3 kanały, każdy z num_bins przedziałami
    std::vector<std::vector<int>> hist(3, std::vector<int>(num_bins, 0));

    if (image.channels() != 3) {
        // Jeśli nie obraz kolorowy, zwróć puste histogramy
        return hist;
    }
    // Obliczanie współczynnika skalowania: mapuje 256 poziomów na num_bins indeksów
    // np. dla num_bins=16, scale = 16.0 / 256.0 = 0.0625
    const double scale = (double)num_bins / 256.0;

    for (int i = 0; i < image.rows; ++i) {
        const cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; ++j) {
            // Obliczamy indeksy dla każdego kanału B, G, R
            int bin_b = (int)(rowPtr[j][0] * scale); 
            int bin_g = (int)(rowPtr[j][1] * scale);
            int bin_r = (int)(rowPtr[j][2] * scale);
            
            // Zabezpieczenie na wypadek błędu zaokrąglenia, aby nie przekroczyć granicy [0, num_bins-1]
            if (bin_b >= num_bins) bin_b = num_bins - 1;
            if (bin_g >= num_bins) bin_g = num_bins - 1;
            if (bin_r >= num_bins) bin_r = num_bins - 1;
            hist[0][bin_b]++; // B
            hist[1][bin_g]++; // G
            hist[2][bin_r]++; // R
        }
    }

    return hist;
}

std::vector<int> calculateCDF(const std::vector<int>& hist) {
    const size_t num_bins = hist.size(); // Pobierz rozmiar z wejściowego histogramu
    std::vector<int> cdf(num_bins);     // Dynamiczna alokacja
    int cumulative = 0;
    
    for (size_t i = 0; i < num_bins; ++i) { // Pętla do num_bins
        cumulative += hist[i];
        cdf[i] = cumulative;
    }
    
    return cdf;
}

cv::Mat applyEqualization(const cv::Mat& inputImage, const std::vector<int>& cdf) {
    // Kopia obrazu wejściowego
    cv::Mat outputImage = inputImage.clone(); 
    const int MAX_LEVELS = 256;
    const size_t num_bins = cdf.size(); // Rozmiar wejściowego CDF
    
    int totalPixels = inputImage.rows * inputImage.cols;
    int L_minus_1 = 255;
    
    // 1. Znalezienie CDF_min (pierwsza niezerowa wartość w CDF)
    int cdf_min = 0;
    for (int val : cdf) {
        if (val > 0) {
            cdf_min = val;
            break;
        }
    }

    // 2. Utworzenie tablicy transformacji (LUT) o rozmiarze 256
    // Wzór: LUT[oryginalny_piksel 0-255] = nowa_wartość
    std::vector<uchar> transformLUT(MAX_LEVELS);
    
    // Współczynnik skalowania: mapuje indeks piksela (0-255) na indeks BINA (0 do num_bins-1)
    const double scale_to_bin = (double)num_bins / (double)MAX_LEVELS;

    for (int i = 0; i < MAX_LEVELS; ++i) { // Pętla na 256 wyjściowych poziomach
        // Który bin odpowiada oryginalnej wartości piksela i?
        int bin_index = (int)(i * scale_to_bin); 
        if (bin_index >= num_bins) bin_index = num_bins - 1;

        if (cdf[bin_index] == 0) {
            transformLUT[i] = 0;
        } else {
            // Używamy wartości z CDF[bin_index] do obliczenia nowej wartości
            float normalized = (float)(cdf[bin_index] - cdf_min) / (totalPixels - cdf_min);
            transformLUT[i] = cv::saturate_cast<uchar>(normalized * L_minus_1);
        }
    }
    
    // 3. Aplikacja transformacji na obrazie (bez zmian, używa 256-elementowego LUT)
    for (int i = 0; i < outputImage.rows; ++i) {
        uchar* rowPtr = outputImage.ptr<uchar>(i);
        for (int j = 0; j < outputImage.cols; ++j) {
            rowPtr[j] = transformLUT[rowPtr[j]];
        }
    }
    
    return outputImage;
}

// Equalizacja każdego kanału kolorowego osobno
cv::Mat applyColorEqualization(const cv::Mat& inputImage, int num_bins) {
    const int MAX_LEVELS = 256;
    
    // Zabezpieczenie argumentu num_bins
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    if (inputImage.channels() != 3) {
        return inputImage.clone(); // Nie kolorowy obraz – zwróć kopię
    }

    std::vector<cv::Mat> channels(3);
    cv::split(inputImage, channels); // Rozdzielenie kanałów B,G,R

    for (int c = 0; c < 3; ++c) {
        std::vector<int> hist = calculateHistogram(channels[c], num_bins); // histogram kanału
        std::vector<int> cdf = calculateCDF(hist);               // CDF kanału
        channels[c] = applyEqualization(channels[c], cdf);       // equalizacja kanału
    }

    cv::Mat outputImage;
    cv::merge(channels, outputImage); // Złączenie kanałów w obraz kolorowy
    return outputImage;
}

cv::Mat equalize_SEQ_Grayscale(const cv::Mat& inputImage, int num_bins) {
    const int MAX_LEVELS = 256;
    
    // Zabezpieczenie argumentu num_bins
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    // Wywołuje kroki sekwencyjne: Histogram -> CDF -> Apply
    std::vector<int> hist = calculateHistogram(inputImage, num_bins);
    std::vector<int> cdf = calculateCDF(hist);
    // Używamy oryginalnej funkcji applyEqualization, która wymaga CDF
    return applyEqualization(inputImage, cdf); 
}

cv::Mat equalize_SEQ_Color(const cv::Mat& inputImage, int num_bins) {
    // Wersja kolorowa już jest zaimplementowana jako applyColorEqualization
    // Zmieniamy tylko nazwę, aby była spójna z nowym wzorcem
    return applyColorEqualization(inputImage, num_bins); 
}
