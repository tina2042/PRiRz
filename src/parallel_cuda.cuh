#pragma once
#include <opencv2/opencv.hpp>

// Deklaracje funkcji C z pliku .cu
extern "C" void computeHistogramCUDA(const unsigned char* input, int* histogram, int width, int height, int threadsPerBlock, int numBlocks, int num_bins);
extern "C" void computeHistogramCUDAColor(const uchar* input, int* histR, int* histG, int* histB, int width, int height, int threadsPerBlock, int numBlocks, int num_bins);
// Wersje wysokopoziomowe – do wywołania z main.cpp
cv::Mat equalize_CUDA_Grayscale(const cv::Mat& inputImage, int threadsPerBlock, int num_bins);
cv::Mat equalize_CUDA_Color(const cv::Mat& inputImage, int threadsPerBlock, int num_bins);