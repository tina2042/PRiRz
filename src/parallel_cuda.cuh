#pragma once
#include <opencv2/opencv.hpp>

// Deklaracja funkcji C z pliku .cu
extern "C" void computeHistogramCUDA(const unsigned char* input, int* histogram, int width, int height);

// Wersja wysokopoziomowa – equalizacja w CUDA (do wywołania z main.cpp)
cv::Mat equalize_CUDA_Grayscale(const cv::Mat& inputImage);
