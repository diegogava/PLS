#pragma once

#include <cuda_runtime.h>

#include "pls.h"

#define BLOCK_SIZE 16

class PLSCuda : public PLS{

public:
	PLSCuda(bool debug);
	void run(cv::Mat feats, cv::Mat labels, const int nfactors);
	void test();

protected:

	void PLSCuda::gpu_FindHighestNormX(cv::Mat X, cv::Mat TempX, double& MaxValX, int& MaxIndexX, float* X_gpu);
	void PLSCuda::gpu_FindHighestNormY(cv::Mat& Y, cv::Mat& TempY, double& MaxValY, int& MaxIndexY, float* Y_gpu);
	void PLSCuda::gpu_SaveResults_tTemp_uTemp(cv::Mat X, cv::Mat Y, cv::Mat tTemp, cv::Mat uTemp, cv::Mat tTemp_TEST, cv::Mat uTemp_TEST, double MaxIndexX, double MaxIndexY, float* X_gpu, float* Y_gpu, float* tTemp_gpu, float* uTemp_gpu);
	void PLSCuda::gpu_SaveResults_T_U (cv::Mat X, cv::Mat T, cv::Mat U, cv::Mat tTemp, cv::Mat uTemp, cv::Mat T_TEST, cv::Mat U_TEST, int index1, int nMaxIterations, float* T_gpu,float* U_gpu,float* tTemp_gpu,float* uTemp_gpu);
	void PLSCuda::gpu_SaveResults_P_W (cv::Mat X, cv::Mat P, cv::Mat W, cv::Mat pTemp, cv::Mat wTemp, cv::Mat P_TEST, cv::Mat W_TEST, int index1, int nMaxIterations, float* P_gpu, float* W_gpu, float* pTemp_gpu, float* wTemp_gpu);
	void PLSCuda::gpu_SaveResults_Q(cv::Mat Y, cv::Mat Q, cv::Mat qTemp, cv::Mat Q_TEST, int index1, int nMaxIterations, float* Q_gpu, float* qTemp_gpu);
	void PLSCuda::gpu_Iterations(cv::Mat X, cv::Mat Y, cv::Mat& tTemp, cv::Mat& uTemp, cv::Mat& wTemp, cv::Mat& qTemp, float *X_gpu, float *Y_gpu, float *tTemp_gpu, float *uTemp_gpu, float *wTemp_gpu, float *qTemp_gpu, float *tNew_gpu, float *sub_gpu, int nMaxOuter);
	void PLSCuda::gpu_Deflation(cv::Mat& X, cv::Mat& Y, cv::Mat& tTemp, cv::Mat& uTemp, cv::Mat& qTemp, cv::Mat& tNorm, cv::Mat& bTemp, cv::Mat& pTemp, float *X_gpu, float *Y_gpu, float *tTemp_gpu, float *uTemp_gpu, float *qTemp_gpu, float *tNorm_gpu, float *bTemp_gpu, float *pTemp_gpu, float *hTemp_gpu, float *gTemp_gpu);
	


	static void CHECK(cudaError_t cudaStatus, std::string prefixMessage);

	void opencvCopy();
	void testDIV();
	void testNORM();
	void testMULTIPLICACAO();
	void testCOPIA();
	void testCOPIACOLUNA();
	void testCOPIAVETORCOLUNA();
	void testSUB();
	void testMULTIPLICACAO_TRANSPOSTA();
	void testAntonio();
	void testMULTIPLICACAO_TRANSPOSTA2();
	void testMULTIPLICAPORESCALAR();
	void testSETAELEMENTO();
	void testNORMAMATRIZ();
	void testFINDHIGHESTNORM();
	void testMULTIPLICAPORPRIMEIROELEMENTO();

};