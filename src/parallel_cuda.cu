#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>

#include "parallel_cuda.cuh"

__device__ uchar __saturate_global_255(float x) {
    if (x < 0.0f) return 0;
    if (x > 255.0f) return 255;
    return (uchar)roundf(x); 
}

__global__ void prefixSumKernel(const int* hist_in, int* cdf_out, int num_bins) {
    
    if (num_bins > 0) {
        cdf_out[0] = hist_in[0];
    }

    for (int i = 1; i < num_bins; ++i) {
        cdf_out[i] = cdf_out[i - 1] + hist_in[i];
    }
}

__global__ void prefixSumColorKernel(const int* histR_in, const int* histG_in, const int* histB_in,
                                     int* cdfR_out, int* cdfG_out, int* cdfB_out, int num_bins) {
    
    if (num_bins > 0) {
        cdfR_out[0] = histR_in[0];
        cdfG_out[0] = histG_in[0];
        cdfB_out[0] = histB_in[0];
    }

    for (int i = 1; i < num_bins; ++i) {
        cdfR_out[i] = cdfR_out[i - 1] + histR_in[i];
        cdfG_out[i] = cdfG_out[i - 1] + histG_in[i];
        cdfB_out[i] = cdfB_out[i - 1] + histB_in[i];
    }
}

__global__ void buildLUTKernel(const int* cdf, uchar* lut_out, int total_pixels, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAX_LEVELS = 256; 

    if (tid >= MAX_LEVELS) return; 

    int cdf_min = 0;
    
    for (int i = 0; i < num_bins; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    if (total_pixels == cdf_min) { 
        lut_out[tid] = tid; 
        return;
    }

    float scale_to_bin = (float)num_bins / (float)MAX_LEVELS;
    int bin_index = (int)(tid * scale_to_bin);
    
    if (bin_index >= num_bins) bin_index = num_bins - 1;

    float cdf_value = (float)cdf[bin_index];
    
    float result = ((cdf_value - (float)cdf_min) / (total_pixels - (float)cdf_min)) * 255.0f;

    lut_out[tid] = (uchar)__saturate_global_255(result); 
}

__global__ void buildLUTColorKernel(const int* cdfR, const int* cdfG, const int* cdfB,
                                    uchar* lutR_out, uchar* lutG_out, uchar* lutB_out, 
                                    int total_pixels, int num_bins) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAX_LEVELS = 256; 

    if (tid >= MAX_LEVELS) return; 

    int cdfR_min = 0, cdfG_min = 0, cdfB_min = 0;
    
    for (int i = 0; i < num_bins; ++i) {
        if (cdfR[i] > 0 && cdfR_min == 0) cdfR_min = cdfR[i];
        if (cdfG[i] > 0 && cdfG_min == 0) cdfG_min = cdfG[i];
        if (cdfB[i] > 0 && cdfB_min == 0) cdfB_min = cdfB[i];
    }
    
    float scale_to_bin = (float)num_bins / (float)MAX_LEVELS;
    int bin_index = (int)(tid * scale_to_bin);
    if (bin_index >= num_bins) bin_index = num_bins - 1;

    float cdfR_val = (float)cdfR[bin_index];
    float cdfG_val = (float)cdfG[bin_index];
    float cdfB_val = (float)cdfB[bin_index];
    
    if (total_pixels != cdfR_min) {
        float resultR = ((cdfR_val - cdfR_min) / (total_pixels - cdfR_min)) * 255.0f;
        lutR_out[tid] = __saturate_global_255(resultR); 
    } else { lutR_out[tid] = tid; }

    if (total_pixels != cdfG_min) {
        float resultG = ((cdfG_val - cdfG_min) / (total_pixels - cdfG_min)) * 255.0f;
        lutG_out[tid] = __saturate_global_255(resultG); 
    } else { lutG_out[tid] = tid; }
    
    if (total_pixels != cdfB_min) {
        float resultB = ((cdfB_val - cdfB_min) / (total_pixels - cdfB_min)) * 255.0f;
        lutB_out[tid] = __saturate_global_255(resultB); 
    } else { lutB_out[tid] = tid; }
}

__global__ void applyLUTKernel(uchar* image_data, const uchar* lut, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        image_data[i] = lut[image_data[i]];
    }
}

__global__ void applyLUTColorKernel(uchar* image_data, const uchar* lutR, const uchar* lutG, const uchar* lutB, int size_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size_pixels; i += stride) {   
        uchar valB = image_data[3 * i + 0];
        uchar valG = image_data[3 * i + 1];
        uchar valR = image_data[3 * i + 2];

        image_data[3 * i + 0] = lutB[valB];
        image_data[3 * i + 1] = lutG[valG];
        image_data[3 * i + 2] = lutR[valR];
    }
}

// Kernel CUDA - histogram grayscale
__global__ void histogramKernel(const unsigned char* image, int* hist, int size, int num_bins, const float scale) {
    extern __shared__ int localHist[]; 
    int tid = threadIdx.x;

    if (tid < num_bins) localHist[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        int pixel_value = image[i];
        
        int bin_index = (int)(pixel_value * scale);
        if (bin_index >= num_bins) bin_index = num_bins - 1;
        
        atomicAdd(&localHist[bin_index], 1);
    }
    __syncthreads();

    if (tid < num_bins) {
        atomicAdd(&hist[tid], localHist[tid]);
    }
}

// Kernel CUDA – histogram dla kolorowego obrazu
__global__ void histogramKernelColor(const uchar* image, int* histR, int* histG, int* histB, int size, int num_bins, const float scale) {
    extern __shared__ int localHist[];
    int *localB = localHist;
    int *localG = localHist + num_bins;
    int *localR = localHist + 2 * num_bins;

    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = tid; i < num_bins * 3; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        uchar b = image[3*i + 0];
        uchar g = image[3*i + 1];
        uchar r = image[3*i + 2];

        int bin_b = (int)(b * scale);
        int bin_g = (int)(g * scale);
        int bin_r = (int)(r * scale);
        
        if (bin_b >= num_bins) bin_b = num_bins - 1;
        if (bin_g >= num_bins) bin_g = num_bins - 1;
        if (bin_r >= num_bins) bin_r = num_bins - 1;

        atomicAdd(&localB[bin_b], 1);
        atomicAdd(&localG[bin_g], 1);
        atomicAdd(&localR[bin_r], 1);
    }
    __syncthreads();

    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&histR[i], localR[i]);
        atomicAdd(&histG[i], localG[i]);
        atomicAdd(&histB[i], localB[i]);
    }
}

extern "C" void computeHistogramCUDAColor(const uchar* input, int* histR, int* histG, int* histB, int width, int height, int threadsPerBlock, int numBlocks, int num_bins) {
    int size = width * height;
    float scale = (float)num_bins / 256.0f; 
    int hist_size_bytes = num_bins * sizeof(int);

    uchar* d_image;
    int *d_histR, *d_histG, *d_histB;

    cudaMalloc(&d_image, size * 3 * sizeof(uchar));
    cudaMalloc(&d_histR, hist_size_bytes); 
    cudaMalloc(&d_histG, hist_size_bytes); 
    cudaMalloc(&d_histB, hist_size_bytes); 

    cudaMemset(d_histR, 0, hist_size_bytes);
    cudaMemset(d_histG, 0, hist_size_bytes);
    cudaMemset(d_histB, 0, hist_size_bytes);

    cudaMemcpy(d_image, input, size * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

    int shared_mem_size = 3 * hist_size_bytes; 
    
    histogramKernelColor<<<numBlocks, threadsPerBlock, shared_mem_size>>>(
        d_image, d_histR, d_histG, d_histB, size, num_bins, scale
    );
    cudaDeviceSynchronize();

    cudaMemcpy(histR, d_histR, hist_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(histG, d_histG, hist_size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(histB, d_histB, hist_size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_histR);
    cudaFree(d_histG);
    cudaFree(d_histB);
}

// CUDA equalization – kolor
cv::Mat equalize_CUDA_Color(const cv::Mat& inputImage, int threadsPerBlock, int num_bins) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int total_pixels = width * height;
    const int MAX_LEVELS = 256;
    
    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));
    std::vector<int> histogramR(num_bins, 0), histogramG(num_bins, 0), histogramB(num_bins, 0);
    int numBlocks = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    
    computeHistogramCUDAColor(inputImage.data, histogramR.data(), histogramG.data(), histogramB.data(), 
                              width, height, threadsPerBlock, numBlocks, num_bins); 
    
    int *d_histR, *d_histG, *d_histB;
    int *d_cdfR, *d_cdfG, *d_cdfB;
    uchar *d_lutR, *d_lutG, *d_lutB;
    uchar *d_image_data; 

    size_t hist_size_bytes = num_bins * sizeof(int);
    size_t lut_size_bytes = MAX_LEVELS * sizeof(uchar);

    cudaMalloc(&d_histR, hist_size_bytes);
    cudaMalloc(&d_histG, hist_size_bytes);
    cudaMalloc(&d_histB, hist_size_bytes);
    
    cudaMalloc(&d_cdfR, hist_size_bytes);
    cudaMalloc(&d_cdfG, hist_size_bytes);
    cudaMalloc(&d_cdfB, hist_size_bytes);

    cudaMalloc(&d_lutR, lut_size_bytes);
    cudaMalloc(&d_lutG, lut_size_bytes);
    cudaMalloc(&d_lutB, lut_size_bytes);
    
    cudaMalloc(&d_image_data, total_pixels * 3 * sizeof(uchar));
    
    cudaMemcpy(d_histR, histogramR.data(), hist_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histG, histogramG.data(), hist_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_histB, histogramB.data(), hist_size_bytes, cudaMemcpyHostToDevice);

    prefixSumColorKernel<<<1, 1>>>(d_histR, d_histG, d_histB, d_cdfR, d_cdfG, d_cdfB, num_bins);
    cudaDeviceSynchronize();

    buildLUTColorKernel<<<1, MAX_LEVELS>>>(d_cdfR, d_cdfG, d_cdfB, 
                                            d_lutR, d_lutG, d_lutB, 
                                            total_pixels, num_bins);
    cudaDeviceSynchronize();

    cv::Mat outputImage(height, width, CV_8UC3);
    
    cudaMemcpy(d_image_data, inputImage.data, total_pixels * 3 * sizeof(uchar), cudaMemcpyHostToDevice);
    
    applyLUTColorKernel<<<numBlocks, threadsPerBlock>>>(d_image_data, d_lutR, d_lutG, d_lutB, total_pixels);
    cudaDeviceSynchronize();
    
    cudaMemcpy(outputImage.data, d_image_data, total_pixels * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_histR); cudaFree(d_histG); cudaFree(d_histB);
    cudaFree(d_cdfR); cudaFree(d_cdfG); cudaFree(d_cdfB);
    cudaFree(d_lutR); cudaFree(d_lutG); cudaFree(d_lutB);
    cudaFree(d_image_data);

    return outputImage;
}

extern "C" void computeHistogramCUDA(const unsigned char* input, int* histogram, int width, int height, int threadsPerBlock, int numBlocks, int num_bins) {
    int size = width * height;
    float scale = (float)num_bins / 256.0f; 
    
    unsigned char* d_image;
    int* d_hist;

    cudaMalloc(&d_image, size * sizeof(unsigned char));
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    cudaMemset(d_hist, 0, num_bins * sizeof(int));

    cudaMemcpy(d_image, input, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    histogramKernel<<<numBlocks, threadsPerBlock, num_bins * sizeof(int)>>>(d_image, d_hist, size, num_bins, scale); 
    cudaDeviceSynchronize();

    cudaMemcpy(histogram, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_hist);
}

cv::Mat equalize_CUDA_Grayscale(const cv::Mat& inputImage, int threadsPerBlock, int num_bins) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int total_pixels = width * height;
    const int MAX_LEVELS = 256; 

    num_bins = std::max(1, std::min(MAX_LEVELS, num_bins));

    std::vector<int> histogram(num_bins, 0);
    int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    
    computeHistogramCUDA(inputImage.data, histogram.data(), width, height, threadsPerBlock, numBlocks, num_bins);

    int* d_hist;
    int* d_cdf;
    uchar* d_lut;
    uchar* d_output_data;
    
    cudaMalloc(&d_hist, num_bins * sizeof(int));
    cudaMalloc(&d_cdf, num_bins * sizeof(int));
    cudaMalloc(&d_lut, MAX_LEVELS * sizeof(uchar));
    
    cudaMemcpy(d_hist, histogram.data(), num_bins * sizeof(int), cudaMemcpyHostToDevice);

    prefixSumKernel<<<1, 1>>>(d_hist, d_cdf, num_bins);
    cudaDeviceSynchronize();

    buildLUTKernel<<<1, MAX_LEVELS>>>(d_cdf, d_lut, total_pixels, num_bins);
    cudaDeviceSynchronize();

    cv::Mat outputImage(height, width, CV_8UC1);
    
    cudaMalloc(&d_output_data, total_pixels * sizeof(uchar));
    
    cudaMemcpy(d_output_data, inputImage.data, total_pixels * sizeof(uchar), cudaMemcpyHostToDevice);
    
    applyLUTKernel<<<numBlocks, threadsPerBlock>>>(d_output_data, d_lut, total_pixels);
    cudaDeviceSynchronize();
    
    cudaMemcpy(outputImage.data, d_output_data, total_pixels * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_hist);
    cudaFree(d_cdf);
    cudaFree(d_lut);
    cudaFree(d_output_data);

    return outputImage;
}