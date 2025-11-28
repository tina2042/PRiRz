#ifndef PARALLEL_MPI_HPP
#define PARALLEL_MPI_HPP

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Główna funkcja wykonująca równoległą equalizację kontrastu za pomocą MPI.
 * @param inputImage Obraz wejściowy (tylko proces 0 używa go do ładowania).
 * @param rank Numer procesu.
 * @param size Całkowita liczba procesów.
 * @param num_bins Liczba przedziałów histogramu
 * @return cv::Mat Zrównoleglony i zequalizowany obraz (tylko na procesie 0).
 */
cv::Mat equalize_MPI_Grayscale(const cv::Mat& inputImage, int rank, int size, int num_bins);

/**
 * @brief Wykonuje równoległą equalizację kontrastu dla obrazu kolorowego (BGR) za pomocą MPI.
 */
cv::Mat equalize_MPI_Color(const cv::Mat& inputImage, int rank, int size, int num_bins);

#endif // PARALLEL_MPI_HPP