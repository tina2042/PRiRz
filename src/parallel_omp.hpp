#ifndef PARALLEL_OMP_HPP
#define PARALLEL_OMP_HPP

#include <opencv2/opencv.hpp>
#include <vector>

// ----------------------------------------------------------------------
// Wersja w skali szarości (Grayscale)
// ----------------------------------------------------------------------

/**
 * @brief Oblicza histogram w skali szarości równolegle przy użyciu OpenMP.
 * Używa lokalnych histogramów wątków i redukcji.
 * @param image Obraz wejściowy (cv::Mat, CV_8UC1).
 * @return std::vector<int> Globalny histogram (256 elementów).
 */
std::vector<int> calculateHistogram_OMP_Grayscale(const cv::Mat& image);

/**
 * @brief Wykonuje pełną equalizację kontrastu dla obrazu szarego z obliczeniem histogramu w OpenMP.
 */
cv::Mat equalize_OMP_Grayscale(const cv::Mat& inputImage);


// ----------------------------------------------------------------------
// Wersja kolorowa (Color - 3 kanały)
// ----------------------------------------------------------------------

/**
 * @brief Oblicza histogramy dla 3 kanałów (B, G, R) równolegle z użyciem OpenMP.
 * @param image Obraz wejściowy (cv::Mat, CV_8UC3).
 * @return std::vector<std::vector<int>> Wektor 3 histogramów (po 256 elementów każdy).
 */
std::vector<std::vector<int>> calculateHistograms_OMP_Color(const cv::Mat& image);

/**
 * @brief Wykonuje pełną equalizację kontrastu dla obrazu kolorowego (po kanałach) z obliczeniem histogramu w OpenMP.
 */
cv::Mat equalize_OMP_Color(const cv::Mat& inputImage);


#endif // PARALLEL_OMP_HPP