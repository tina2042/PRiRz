#pragma once
#include <opencv2/opencv.hpp>

// Deklaracje funkcji C z pliku .cu
extern "C" void computeHistogramCUDA(const unsigned char* input, int* histogram, int width, int height, int threadsPerBlock, int numBlocks);
extern "C" void computeHistogramCUDAColor(const unsigned char* input, int* histR, int* histG, int* histB, int width, int height, int threadsPerBlock, int numBlocks);

// Wersje wysokopoziomowe – do wywołania z main.cpp
cv::Mat equalize_CUDA_Grayscale(const cv::Mat& inputImage, int threadsPerBlock);
cv::Mat equalize_CUDA_Color(const cv::Mat& inputImage, int threadsPerBlock);
