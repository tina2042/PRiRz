#include <opencv2/opencv.hpp>
#include <vector>

std::vector<int> calculateHistogram(const cv::Mat& image) {
    // Histogram 256 przedziałów (dla 8-bitowego obrazu)
    std::vector<int> hist(256, 0); 
    
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
            hist[pixelValue]++;
        }
    }
    
    return hist;
}

std::vector<int> calculateCDF(const std::vector<int>& hist) {
    std::vector<int> cdf(256);
    int cumulative = 0;
    
    for (int i = 0; i < 256; ++i) {
        cumulative += hist[i];
        cdf[i] = cumulative;
    }
    
    return cdf;
}

cv::Mat applyEqualization(const cv::Mat& inputImage, const std::vector<int>& cdf) {
    // Kopia obrazu wejściowego
    cv::Mat outputImage = inputImage.clone(); 
    
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

    // 2. Utworzenie tablicy transformacji (LUT)
    std::vector<uchar> transformLUT(256);
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] == 0) {
            // Piksele, które się nie pojawiły, mapujemy na 0
            transformLUT[i] = 0;
        } else {
            // Wzór equalizacji
            float normalized = (float)(cdf[i] - cdf_min) / (totalPixels - cdf_min);
            transformLUT[i] = cv::saturate_cast<uchar>(normalized * L_minus_1);
        }
    }
    
    // 3. Aplikacja transformacji na obrazie
    for (int i = 0; i < outputImage.rows; ++i) {
        uchar* rowPtr = outputImage.ptr<uchar>(i);
        for (int j = 0; j < outputImage.cols; ++j) {
            rowPtr[j] = transformLUT[rowPtr[j]];
        }
    }
    
    return outputImage;
}