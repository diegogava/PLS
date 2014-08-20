#include "PLSCublas.h"

#include <cublas.h>

PLSCublas::PLSCublas(bool debug)
	: PLS(debug){

}

void PLSCublas::run(cv::Mat feats, cv::Mat labels, const int nfactors){
	
	cv::Mat gpuX = feats.clone();
	int gpuN = gpuX.rows;
	int gpud = gpuX.cols;
	cv::Mat gpuY = labels.clone();
	int gpuf = gpuY.cols;
	cv::Mat gpuT, gpuP, gpuW, gpuB, gpuXmean, gpuXstd;
	gpuT.create(gpuN, nfactors, CV_32F);
	gpuP.create(gpud, nfactors, CV_32F);
	gpuW.create(gpud, nfactors, CV_32F);
	gpuXmean.create(1, gpud, CV_32F);
	gpuXstd.create(1, gpud, CV_32F);
	gpuB.create(1, nfactors, CV_32F);


	gpuNIPALS(&gpuX.at<float>(0), gpuN, gpud, &gpuY.at<float>(0),
		gpuf, nfactors, &gpuT.at<float>(0), &gpuP.at<float>(0),
		&gpuW.at<float>(0), &gpuB.at<float>(0),
		&gpuXmean.at<float>(0), &gpuXstd.at<float>(0));

}

void PLSCublas::cublasSnormalize(int N, int d, float *d_X, float *mean, float *std) {

	float *temp = new float[N];
	for (int i = 0; i < N; i++)
		temp[i] = 1.0;

	float *d_mean, *d_ones;

	cublasAlloc(N, sizeof (d_ones[0]), (void**) &d_ones);
	cublasAlloc(d, sizeof (d_mean[0]), (void**) &d_mean);

	cublasSetVector(N, sizeof (temp[0]), temp, 1, d_ones, 1);
	cublasSgemv('T', N, d, (float) 1.0 / (float) N, d_X, N, d_ones, 1, 0.0, d_mean, 1);
	cublasGetVector(d, sizeof (temp[0]), d_mean, 1, mean, 1);

	cublasSgemm('N', 'T', N, d, 1, -1.0, d_ones, N, d_mean, d, 1.0, d_X, N);

	for (int i = 0; i < d; i++) {
		float s = (float) sqrt(cublasSdot(N, d_X + i*N, 1, d_X + i*N, 1) / (float) N);
		std[i] = s > 0.001 ? s : 1;
		cublasSscal(N, 1 / std[i], d_X + i*N, 1);
	}

	cublasFree(d_mean);
	cublasFree(d_ones);

	delete [] temp;
}

void PLSCublas::gpuNIPALS(float *X, int N, int d, float *Y, int f, int numFactor, float *T, float *P, float *W, float *b, float *meanX, float *stdX) {

	cublasStatus status;
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Cublas initialization error\n");
		return;
	}

	float *d_X = 0;
	float *d_Y = 0;

	status = cublasAlloc(d*N, sizeof (X[0]), (void**) &d_X);
	status = cublasSetVector(N*d, sizeof (X[0]), X, 1, d_X, 1);
	status = cublasAlloc(f*N, sizeof (Y[0]), (void**) &d_Y);
	status = cublasSetVector(N*f, sizeof (Y[0]), Y, 1, d_Y, 1);

	cublasSnormalize(N, d, d_X, meanX, stdX);

	float *meanY = new float[f];
	float *stdY = new float[f];
	cublasSnormalize(N, f, d_Y, meanY, stdY);
	delete [] meanY;
	delete [] stdY;

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Cublas error in X/Y\n");
		return;
	}

	float *d_t = 0;
	float *d_t0 = 0;
	float *d_w = 0;
	float *d_c = 0;
	float *d_u = 0;
	float *d_p = 0;

	status = cublasAlloc(N, sizeof (d_t[0]), (void**) &d_t);
	status = cublasAlloc(N, sizeof (d_t0[0]), (void**) &d_t0);
	status = cublasAlloc(d, sizeof (d_w[0]), (void**) &d_w);
	status = cublasAlloc(f, sizeof (d_c[0]), (void**) &d_c);
	status = cublasAlloc(N, sizeof (d_u[0]), (void**) &d_u);
	status = cublasAlloc(d, sizeof (d_p[0]), (void**) &d_p);

	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Cublas error in allocating memory to vectors\n");
		return;
	}

	for (int i = 0; i < numFactor; i++) {
		float eps = (float) 1e-6;
		cublasScopy(N, d_Y, 1, d_t, 1);
		cublasSscal(N, 1 / cublasSnrm2(N, d_t, 1), d_t, 1);
		cublasScopy(N, d_t, 1, d_u, 1);

		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("Cublas error in initializing iteration vectors\n");
			return;
		}
		for (int j = 0; j < 1; j++) {
			cublasScopy(N, d_t, 1, d_t0, 1);

			// w=normalize(X'*u)
			cublasSgemv('T', N, d, 1.0, d_X, N, d_u, 1, 0.0, d_w, 1);
			cublasSscal(d, 1 / cublasSnrm2(d, d_w, 1), d_w, 1);
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS) {
				printf("Cublas error in X'*u\n");
				return;
			}

			// t=normalize(X*w)
			cublasSgemv('N', N, d, 1.0, d_X, N, d_w, 1, 0.0, d_t, 1);
			cublasSscal(N, 1 / cublasSnrm2(N, d_t, 1), d_t, 1);
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS) {
				printf("Cublas error in X*w\n");
				return;
			}

			//c=normalize(Y'*t)            
			cublasSgemv('T', N, f, 1.0, d_Y, N, d_t, 1, 0.0, d_c, 1);
			cublasSscal(f, 1 / cublasSnrm2(f, d_c, 1), d_c, 1);
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS) {
				printf("Cublas error in Y'*w\n");
				return;
			}

			//u=(Y*c)
			cublasSgemv('N', N, f, 1.0, d_Y, N, d_c, 1, 0.0, d_u, 1);
			//cublasSscal(N,1/cublasSnrm2(N,d_u,1),d_u,1);	
			status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS) {
				printf("Cublas error in Y*t\n");
				return;
			}

			cublasSaxpy(N, -1.0, d_t, 1, d_t0, 1);
			float check = cublasSdot(N, d_t0, 1, d_t0, 1);
			if (check <= eps / 2.0)
				break;

		}
		// p=(X'*t)
		cublasSgemv('T', N, d, 1.0, d_X, N, d_t, 1, 0.0, d_p, 1);
		b[i] = cublasSdot(N, d_u, 1, d_t, 1) / cublasSdot(N, d_t, 1, d_t, 1);

		// X&Y deflation
		cublasSgemm('N', 'T', N, d, 1, -1.0, d_t, N, d_p, d, 1.0, d_X, N);
		cublasSgemm('N', 'T', N, f, 1, (float) (-1.0 * b[i]), d_t, N, d_c, d, 1.0, d_Y, N);

		status = cublasGetError();
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("Cublas error in X-Y deflation\n");
			return;
		}
		status = cublasGetVector(N, sizeof (d_t[0]), d_t, 1, T + i*N, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("Cublas error in X-Y deflation\n");
			return;
		}

		status = cublasGetVector(d, sizeof (d_p[0]), d_p, 1, P + i*d, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("Cublas error in X-Y deflation\n");
			return;
		}

		status = cublasGetVector(d, sizeof (d_w[0]), d_w, 1, W + i*d, 1);
		status = cublasGetError();
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("Cublas error in X-Y deflation\n");
			return;
		}

		status = cublasGetError();
		if (status != CUBLAS_STATUS_SUCCESS) {
			printf("Cublas error in reading back GPU vector\n");
			return;
		}
	}

	status = cublasFree(d_X);
	status = cublasFree(d_Y);
	status = cublasFree(d_t);
	status = cublasFree(d_t0);
	status = cublasFree(d_w);
	status = cublasFree(d_c);
	status = cublasFree(d_u);
	status = cublasFree(d_p);
	status = cublasShutdown();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Error shutting down Cublas\n");
		return;
	}
}