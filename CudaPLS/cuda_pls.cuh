#ifndef CUDA_PLS_H_
#define CUDA_PLS_H_

#include <cublas.h>
#include <fstream>
#include <unordered_set>
#include <opencv2/opencv.hpp>


class CUDA_PLS {
public :
	void learn(const cv::Mat &_X, const cv::Mat &Y, const int factors, int max_factors = 1);
	static void test();

private:
	cv::Mat nipals(cv::Mat X, cv::Mat Y, const int nfactors);
	void CUDA_PLS::gpu_iterations(cv::Mat X, cv::Mat Y, cv::Mat *tTemp, cv::Mat *uTemp, cv::Mat *wTemp, cv::Mat *qTemp, int nMaxOuter, double TermCrit);
	double normalizeGPU(cv::Mat A);
	cv::Mat subGPU(float *A, float *B, int A_size, int B_size);
}; 


#endif // !CUDA_PLS_H_
