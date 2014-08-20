#ifndef _UTIL_H_
#define	_UTIL_H_

#include <chrono>
#include <opencv2/opencv.hpp>
#include <fstream>

/*
 * Util
 */
inline void cbind(cv::Mat &dst, const cv::Mat src) {
    if (dst.empty()) dst = src;
    else cv::hconcat(src, dst, dst);
}

inline void rbind(cv::Mat &dst, const cv::Mat src) {
    if (dst.empty()) dst = src;
    else cv::vconcat(src, dst, dst);
}

inline void ts_cbind(cv::Mat &dst, const cv::Mat src) {
#pragma omp critical
    {
        cbind(dst, src);
    }
}

inline void ts_rbind(cv::Mat &dst, const cv::Mat src) {
#pragma omp critical
    {
        rbind(dst, src);
    }
}

class counter_t {
    std::chrono::system_clock::time_point ini;
    std::chrono::system_clock::time_point end;
public:

    void tick() {
        ini = std::chrono::high_resolution_clock::now();
    }

    unsigned long long tock() {
        end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - ini);
        return time.count();
    }
};

#endif