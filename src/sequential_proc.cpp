#include <opencv2/opencv.hpp>
#include <vector>

std::vector<int> calculateHistogram(const cv::Mat& image, int num_bins) {
    std::vector<int> hist(num_bins, 0);
    double scale = (double)num_bins / 256.0;
    
    if (image.channels() != 1) {
        return hist; 
    }
    
    for (int i = 0; i < image.rows; ++i) {
        const uchar* rowPtr = image.ptr<uchar>(i);
        for (int j = 0; j < image.cols; ++j) {
            int pixelValue = rowPtr[j];
            int bin_index = (int)(pixelValue * scale); 
            
            if (bin_index >= num_bins) bin_index = num_bins - 1; 
            
            hist[bin_index]++;
        }
    }
    
    return hist;
}

std::vector<std::vector<int>> calculateColorHistogram(const cv::Mat& image, int num_bins) {
    if (num_bins < 1 || num_bins > 256) {
        num_bins = 256; 
    }
    std::vector<std::vector<int>> hist(3, std::vector<int>(num_bins, 0));

    if (image.channels() != 3) {
        return hist;
    }

    const double scale = (double)num_bins / 256.0;

    for (int i = 0; i < image.rows; ++i) {
        const cv::Vec3b* rowPtr = image.ptr<cv::Vec3b>(i);
        for (int j = 0; j < image.cols; ++j) {
            int bin_b = (int)(rowPtr[j][0] * scale); 
            int bin_g = (int)(rowPtr[j][1] * scale);
            int bin_r = (int)(rowPtr[j][2] * scale);
            
            if (bin_b >= num_bins) bin_b = num_bins - 1;
            if (bin_g >= num_bins) bin_g = num_bins - 1;
            if (bin_r >= num_bins) bin_r = num_bins - 1;
            hist[0][bin_b]++; 
            hist[1][bin_g]++; 
            hist[2][bin_r]++; 
        }
    }

    return hist;
}

std::vector<int> calculateCDF(const std::vector<int>& hist) {
    const size_t num_bins = hist.size(); 
    std::vector<int> cdf(num_bins);     
    int cumulative = 0;
    
    for (size_t i = 0; i < num_bins; ++i) { 
        cumulative += hist[i];
        cdf[i] = cumulative;
    }
    
    return cdf;
}

cv::Mat applyEqualization(const cv::Mat& inputImage, const std::vector<int>& cdf) {
    cv::Mat outputImage = inputImage.clone(); 
    const int MAX_LEVELS = 256;
    const size_t num_bins = cdf.size(); 
    
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
    std::vector<uchar> transformLUT(MAX_LEVELS);
    
    const double scale_to_bin = (double)num_bins / (double)MAX_LEVELS;

    for (int i = 0; i < MAX_LEVELS; ++i) { 
        int bin_index = (int)(i * scale_to_bin); 
        if (bin_index >= num_bins) bin_index = num_bins - 1;

        if (cdf[bin_index] == 0) {
            transformLUT[i] = 0;
        } else {
            float normalized = (float)(cdf[bin_index] - cdf_min) / (totalPixels - cdf_min);
            transformLUT[i] = cv::saturate_cast<uchar>(normalized * L_minus_1);
        }
    }
    
    // 3. Aplikacja transformacji na obrazie (używa 256-elementowego LUT)
    for (int i = 0; i < outputImage.rows; ++i) {
        uchar* rowPtr = outputImage.ptr<uchar>(i);
        for (int j = 0; j < outputImage.cols; ++j) {
            rowPtr[j] = transformLUT[rowPtr[j]];
        }
    }
    
    return outputImage;
}

cv::Mat applyColorEqualization(const cv::Mat& inputImage, int num_bins) {
    const int MAX_LEVELS = 256;
    
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    if (inputImage.channels() != 3) {
        return inputImage.clone(); 
    }

    std::vector<cv::Mat> channels(3);
    cv::split(inputImage, channels); 

    for (int c = 0; c < 3; ++c) {
        std::vector<int> hist = calculateHistogram(channels[c], num_bins); 
        std::vector<int> cdf = calculateCDF(hist);               
        channels[c] = applyEqualization(channels[c], cdf);       
    }

    cv::Mat outputImage;
    cv::merge(channels, outputImage); 
    return outputImage;
}

cv::Mat equalize_SEQ_Grayscale(const cv::Mat& inputImage, int num_bins) {
    const int MAX_LEVELS = 256;
    
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    std::vector<int> hist = calculateHistogram(inputImage, num_bins);
    std::vector<int> cdf = calculateCDF(hist);
    return applyEqualization(inputImage, cdf); 
}

cv::Mat equalize_SEQ_Color(const cv::Mat& inputImage, int num_bins) {
    return applyColorEqualization(inputImage, num_bins); 
}
