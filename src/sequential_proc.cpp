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


// Zliczanie histogramu dla obrazów kolorowych (BGR)
std::vector<std::vector<int>> calculateColorHistogram(const cv::Mat& image) {
    std::vector<std::vector<int>> hist(3, std::vector<int>(256, 0)); // 3 kanały: B, G, R

    if (image.channels() != 3) {
        // Jeśli nie obraz kolorowy, zwróć puste histogramy
        return hist;
    }

    for (int i = 0; i < image.rows; ++i) {
        const cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; ++j) {
            hist[0][rowPtr[j][0]]++; // B
            hist[1][rowPtr[j][1]]++; // G
            hist[2][rowPtr[j][2]]++; // R
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

// Equalizacja każdego kanału kolorowego osobno
cv::Mat applyColorEqualization(const cv::Mat& inputImage) {
    if (inputImage.channels() != 3) {
        return inputImage.clone(); // Nie kolorowy obraz – zwróć kopię
    }

    std::vector<cv::Mat> channels(3);
    cv::split(inputImage, channels); // Rozdzielenie kanałów B,G,R

    for (int c = 0; c < 3; ++c) {
        std::vector<int> hist = calculateHistogram(channels[c]); // histogram kanału
        std::vector<int> cdf = calculateCDF(hist);               // CDF kanału
        channels[c] = applyEqualization(channels[c], cdf);       // equalizacja kanału
    }

    cv::Mat outputImage;
    cv::merge(channels, outputImage); // Złączenie kanałów w obraz kolorowy
    return outputImage;
}
