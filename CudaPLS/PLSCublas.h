#pragma once

#include "pls.h"

class PLSCublas : public PLS{

public:
	PLSCublas(bool debug);
	void run(cv::Mat feats, cv::Mat labels, const int nfactors);

private:
	void cublasSnormalize(int N, int d, float *d_X, float *mean, float *std);
	void gpuNIPALS(float *X, int N, int d, float *Y, int f, int numFactor, float *T, float *P, float *W, float *b, float *meanX, float *stdX);

};

