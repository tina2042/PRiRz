#ifndef SEQUENTIAL_PROC_HPP
#define SEQUENTIAL_PROC_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

/**
 * @brief Oblicza histogram jasności dla obrazu w skali szarości.
 * * Histogram to wektor zliczający liczbę pikseli dla każdego z 256 poziomów jasności.
 * * @param image Obraz wejściowy (cv::Mat, oczekiwany CV_8UC1).
 * @return std::vector<int> Histogram (256 elementów).
 */
std::vector<int> calculateHistogram(const cv::Mat& image);

/**
 * @brief Oblicza histogram dla obrazu kolorowego (BGR).
 * Dla każdego kanału (B, G, R) generowany jest osobny histogram.
 * @param image Obraz wejściowy (cv::Mat, oczekiwany CV_8UC3).
 * @return std::vector<std::vector<int>> Wektor 3 histogramów po 256 elementów (kolejno B, G, R).
 */
std::vector<std::vector<int>> calculateColorHistogram(const cv::Mat& image);

/**
 * @brief Oblicza Skumulowaną Dystrybuantę (CDF) na podstawie histogramu.
 * * CDF[i] jest sumą wszystkich wartości histogramu od indeksu 0 do i.
 * * @param hist Wektor histogramu (256 elementów).
 * @return std::vector<int> Skumulowana Dystrybuanta (256 elementów).
 */
std::vector<int> calculateCDF(const std::vector<int>& hist);

/**
 * @brief Stosuje transformację equalizacji kontrastu do obrazu w skali szarości.
 * * Używa CDF do wygenerowania tablicy transformacji (LUT) i mapuje piksele 
 * stare na nowe wartości.
 * * @param inputImage Obraz wejściowy w skali szarości.
 * @param cdf Wektor Skumulowanej Dystrybuanty.
 * @return cv::Mat Nowy obraz po equalizacji.
 */
cv::Mat applyEqualization(const cv::Mat& inputImage, const std::vector<int>& cdf);

/**
 * @brief Equalizuje kontrast dla każdego kanału obrazu kolorowego niezależnie.
 * @param inputImage Obraz wejściowy (CV_8UC3).
 * @return cv::Mat Obraz po equalizacji każdego kanału (B, G, R).
 */
cv::Mat applyColorEqualization(const cv::Mat& inputImage);

/**
 * @brief PEŁNA sekwencyjna equalizacja (wzorzec). 
 * Wykonuje: Histogram -> CDF -> Transformacja.
 */
cv::Mat equalize_SEQ_Grayscale(const cv::Mat& inputImage);

/**
 * @brief PEŁNA sekwencyjna equalizacja kolorów (wzorzec).
 * Wykonuje: Histogramy_kolorów -> CDFs -> Transformacja_kolorów.
 */
cv::Mat equalize_SEQ_Color(const cv::Mat& inputImage);

#endif // SEQUENTIAL_PROC_HPP