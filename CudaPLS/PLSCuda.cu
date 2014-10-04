#include "PLSCuda.cuh"

#include <cuda.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <numeric>
#include <typeinfo>
#include <iomanip>
#include "util.h"

#ifndef __CUDACC__  
#define __CUDACC__
#endif

inline void round_4(double *number){

	//*number = floor( *number*pow(10, 3) - 0.5 )/pow(10, 3); 
	((*number*10000 - (int)(*number*10000))) >= 0.5 ? *number = int((*number + 0.0001)*10000)/10000.0 : *number = int(*number*10000)/10000.0;
	
}

void PLSCuda::CHECK(cudaError_t cudaStatus, std::string prefixMessage){
	if (cudaStatus != cudaSuccess) {
		std::cout << prefixMessage << ": " << cudaGetErrorString(cudaStatus) << std::endl;
		system("pause");
		exit(2);
	}
}

__device__ double normDevice;
__device__ double auxFirstElement = -1000;
__device__ double MaxValXDevice = -10;
__device__ double MaxValYDevice = -10;
__device__ int MaxIndexXDevice = -10;
__device__ float MaxIndexYDevice = -10;


__global__ void mulCUDA(float *A, float *B, float *C, int Arows, int Acols, int Brows, int Bcols);
__global__ void mulTransbyNormalCUDA(float *A_t, float *B, float *C, int A_trows, int A_tcols, int Brows, int Bcols);
__global__ void mulNormalbyTransCUDA (float *A, float *B_t, float *C, int Arows, int Acols, int B_trows, int B_tcols);
__global__ void copyVectorCUDA(float *A, float *B, int size);
__global__ void subCUDA(float *A, float *B, float *C, int cols, int size);
__global__ void divVectorbyNormCUDA(float *A, int size);
__global__ void divVectorbyScalarCUDA(float *A, int size, float value);
__global__ void mulMatrixbyScalarCUDA(float *A, int rows, int cols, float scalar);
__global__ void normCUDA(float *A, int size);
__global__ void copyColumnToVectorCUDA(float *A, float *B, int num_rows, int num_cols, int column);
__global__ void copyVectorToColumnCUDA(float *A, float *B, int num_rows, int num_cols, int column);
__global__ void findhighestXNORMCUDA (float *X, int Xrows, int Xcols);
__global__ void findhighestYNORMCUDA (float *Y, int Yrows, int Ycols);
__global__ void multiplyMatrixbyElementCUDA (float *A, float*B, int Arows, int Acols);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CLASS IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

PLSCuda::PLSCuda(bool debug)
	: PLS(debug){

}

void PLSCuda::run(cv::Mat feats, cv::Mat labels, const int nfactors){

	cv::Mat T, P, U, Q, W, B, _Y;

	cv::Mat Y = labels.clone();
	cv::Mat X = feats.clone();

	// calculate mean and std from X
	cv::Mat Xavg, Xstd;
	Xavg.create(1, X.cols, CV_32F);
	Xstd.create(1, X.cols, CV_32F);

	for (int col = 0; col < X.cols; ++col) {
		cv::Mat avg, std;
		cv::meanStdDev(X.col(col), avg, std);
		Xavg.ptr<float>(0)[col] = *avg.ptr<double>(0);
		Xstd.ptr<float>(0)[col] = *std.ptr<double>(0);
		// normalize X column
		for (int row = 0; row < X.rows; ++row) {
			X.at<float>(row, col) -= Xavg.ptr<float>(0)[col];
			X.at<float>(row, col) /= Xstd.ptr<float>(0)[col];
		}
	}

	// preserve Y
	_Y = Y.clone();

	//Setting the termination criteria
	cv::Mat tNorm;	
	cv::Mat TempX, TempY;
	double MaxValX, MaxValY;
	double normX, normY;
	int nMaxIterations, nMaxOuter = 1000;
	int MaxIndexX, MaxIndexY;

	nMaxIterations = nfactors;	
	
	//Matrices for storing the intermediate values.
	cv::Mat tTemp, tNew, uTemp, wTemp, qTemp, pTemp, bTemp;
	cv::Mat TempX_TEST, TempY_TEST, tTemp_TEST, uTemp_TEST, T_TEST, U_TEST, P_TEST, W_TEST, Q_TEST, B_TEST;
	float *X_gpu, *Y_gpu, *tTemp_gpu, *uTemp_gpu, *wTemp_gpu, *qTemp_gpu;
	float *tNew_gpu, *sub_gpu;
	float *tNorm_gpu, *bTemp_gpu, *pTemp_gpu, *hTemp_gpu, *gTemp_gpu;
	float *TempX_gpu, *TempY_gpu, *T_gpu, *U_gpu, *P_gpu, *W_gpu, *Q_gpu, *B_gpu;
	
	//Allocating memory for TESTS
	tTemp_TEST.create(X.rows, 1, CV_32F);
	uTemp_TEST.create(Y.rows, 1, CV_32F);
	T_TEST.create(X.rows, nMaxIterations, CV_32F);
	U_TEST.create(Y.rows, nMaxIterations, CV_32F);
	P_TEST.create(X.cols, nMaxIterations, CV_32F);
	W_TEST.create(X.cols, nMaxIterations, CV_32F);
	Q_TEST.create(X.cols, nMaxIterations, CV_32F);
	B_TEST.create(nMaxIterations, nMaxIterations, CV_32F);

	//Allocating memory
	T.create(X.rows, nMaxIterations, CV_32F); // create(nrows, ncols, type)
	P.create(X.cols, nMaxIterations, CV_32F);
	U.create(Y.rows, nMaxIterations, CV_32F);
	Q.create(Y.cols, nMaxIterations, CV_32F);
	W.create(X.cols, nMaxIterations, CV_32F);
	B.create(nMaxIterations, nMaxIterations, CV_32F);
	tTemp.create(X.rows, 1, CV_32F);
	uTemp.create(Y.rows, 1, CV_32F);
	wTemp.create(X.cols, 1, CV_32F);
	qTemp.create(Y.cols, 1, CV_32F);
	tNorm.create(1, 1, CV_32F);
	bTemp.create(1, 1, CV_32F);
	pTemp.create(X.cols,1,CV_32F);

	/* alocando memoria global da GPU */
	cudaMalloc((void **) &X_gpu, X.rows * X.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (1)");
	cudaMalloc((void **) &Y_gpu, Y.rows * Y.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (2)");

	//Iteration
	cudaMalloc((void **) &tTemp_gpu, tTemp.rows * tTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (3)");
	cudaMalloc((void **) &uTemp_gpu, uTemp.rows * uTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (4)");
	cudaMalloc((void **) &wTemp_gpu, wTemp.rows * wTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (5)");
	cudaMalloc((void **) &qTemp_gpu, qTemp.rows * qTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (6)");
	cudaMalloc((void **) &tNew_gpu, X.rows * 1 * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (7)");
	cudaMalloc((void **) &sub_gpu, X.rows * 1 * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (8)");

	//Deflation
	cudaMalloc((void **) &tNorm_gpu, tTemp.cols * tTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (9)");
	cudaMalloc((void **) &bTemp_gpu, uTemp.cols * tTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (10)");
	cudaMalloc((void **) &pTemp_gpu, X.cols * tTemp.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (11)");
	cudaMalloc((void **) &hTemp_gpu, tTemp.rows * X.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (12)");
	cudaMalloc((void **) &gTemp_gpu, tTemp.rows * qTemp.rows * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (13)");
	
	
	// Find Highest Norm	
	cudaMalloc((void **) &TempX_gpu, X.rows * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (14)");
	cudaMalloc((void **) &TempY_gpu, Y.rows * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (15)");
	
	// Copies
	cudaMalloc((void **) &T_gpu, X.rows * nMaxIterations * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (16)");
	cudaMalloc((void **) &U_gpu, Y.rows * nMaxIterations * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (17)");
	cudaMalloc((void **) &P_gpu, X.cols * nMaxIterations * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (18)");
	cudaMalloc((void **) &W_gpu, X.cols * nMaxIterations * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (19)");
	cudaMalloc((void **) &Q_gpu, Y.cols * nMaxIterations * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (20)");
	cudaMalloc((void **) &B_gpu, nMaxIterations * nMaxIterations * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (21)");

	/* Copia dados da RAM para a memoria global da GPU */
	cudaMemcpy(X_gpu, &X.at<float>(0), X.rows * X.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device (1)");
	cudaMemcpy(Y_gpu, &Y.at<float>(0), Y.rows * Y.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device (2)");
	
	std::cout << " Versao Paralela " << std::endl;
	tic("Total");
	for (int index1 = 0; index1 < nMaxIterations; index1++) {
		std::cout << " ITERACAO DO LACO MAIOR: " << index1 << std::endl;
		//Finding the column having the highest norm
		MaxValX = 0;
		MaxValY = 0;
		MaxIndexX = -10;
		MaxIndexY = -10;
		
		TempX.create(X.rows, 1, X.type());
		TempY.create(Y.rows, 1, Y.type());

		
		tic("Acha maior norma X");gpu_FindHighestNormX (X, TempX, MaxValX, MaxIndexX, X_gpu);tac("Acha maior norma X");
			
		tic("Acha maior norma Y");gpu_FindHighestNormY (Y, TempY, MaxValY, MaxIndexY, Y_gpu);tac("Acha maior norma Y");
			
		for (int index3 = 0; index3 < X.rows; index3++) {

			tTemp.at<float>(index3, 0) = X.at<float>(index3, MaxIndexX);
			uTemp.at<float>(index3, 0) = Y.at<float>(index3, MaxIndexY);

		}
		
		tic("Salva Resultados em tTemp e uTemp");gpu_SaveResults_tTemp_uTemp (X, Y, tTemp, uTemp, tTemp_TEST, uTemp_TEST, MaxIndexX, MaxIndexY, X_gpu, Y_gpu, tTemp_gpu, uTemp_gpu ); tac("Salva Resultados em tTemp e uTemp");

		//gpu iteration
		tic("GPU Iterations"); gpu_Iterations(X, Y, tTemp, uTemp, wTemp, qTemp, X_gpu, Y_gpu, tTemp_gpu, uTemp_gpu, wTemp_gpu, qTemp_gpu, tNew_gpu, sub_gpu, nMaxOuter); tac("GPU Iterations");

		//gpu_deflation
		tic("GPU Deflation"); gpu_Deflation(X, Y, tTemp, uTemp, qTemp, tNorm, bTemp, pTemp, X_gpu, Y_gpu, tTemp_gpu, uTemp_gpu, tNorm_gpu, qTemp_gpu, bTemp_gpu, pTemp_gpu, hTemp_gpu, gTemp_gpu); tac("GPU Deflation");
		
		

		// Saving Results to Outputs.
		for (int index3 = 0; index3 != X.rows; index3++) {
			T.at<float>(index3, index1) = tTemp.at<float>(index3, 0);
			U.at<float>(index3, index1) = uTemp.at<float>(index3, 0);			
		}

		tic("Salva Resultados em T e U"); gpu_SaveResults_T_U (X, T, U, tTemp, uTemp, T_TEST, U_TEST, index1, nMaxIterations, T_gpu, U_gpu, tTemp_gpu, uTemp_gpu );tac("Salva Resultados em T e U");

		for (int index3 = 0; index3 != X.cols; index3++) {
			P.at<float>(index3, index1) = pTemp.at<float>(index3, 0);
			W.at<float>(index3, index1) = wTemp.at<float>(index3, 0);			
		}

		tic("Salva Resultados em P e W"); gpu_SaveResults_P_W (X, P, W, pTemp, wTemp, Q_TEST,  W_TEST, index1, nMaxIterations, P_gpu, W_gpu, pTemp_gpu, wTemp_gpu );tac("Salva Resultados em P e W");

		for (int index3 = 0; index3 != qTemp.rows; index3++) {
			Q.at<float>(index3, index1) = qTemp.at<float>(index3, 0);	
		}

		tic("Salva Resultados em Q"); gpu_SaveResults_Q (Y, Q, qTemp, Q_TEST, index1, nMaxIterations, Q_gpu, qTemp_gpu); tac("Salva Resultados em Q");

		B.at<float>(index1, index1) = bTemp.at<float>(0, 0);

		tic("Copia resultado para B"); cudaMemcpy(&B_gpu[index1*nMaxIterations + index1], &bTemp_gpu[0], sizeof(float),cudaMemcpyDeviceToDevice); CHECK(cudaGetLastError(), "Cópia do Device para o Device"); tac("Copia resultado para B");
		

		//B.at<float>(index1, index1) = bTemp.at<float>(0, 0); 

		/*std::cout << "TESTANDO (B.at<float>(index1, index1) = bTemp.at<float>(0, 0)) " << std::endl;
		cudaMemcpy(&B_TEST.at<float>(0), B_gpu, nMaxIterations * nMaxIterations * sizeof(float) , cudaMemcpyDeviceToHost);	
		std::cout << " Original = " << B.at<float>(index1,index1) << " GPU = " << B_TEST.at<float>(index1,index1) << std::endl;
		system("pause");*/

		tic("Teste (normX == 0) || (normY == 0) ");
		normCUDA<<< 1, 1>>>(X_gpu, X.rows*X.cols); CHECK(cudaGetLastError(), "Calcula Norma (1)");
		cudaMemcpyFromSymbol(&normX, normDevice, sizeof(normX), 0, cudaMemcpyDeviceToHost); 
		
		/*std::cout << "TESTANDO ( cv::norm(X) == 0 ) " << std::endl;
		std::cout << "\n Norma CPU = " << cv::norm(X) << " Norma GPU = " << normX << std::endl;
		system("pause");*/

		normCUDA<<< 1, 1>>>(Y_gpu, Y.rows*Y.cols); CHECK(cudaGetLastError(), "Calcula Norma (2)");
		cudaMemcpyFromSymbol(&normY, normDevice, sizeof(normY), 0, cudaMemcpyDeviceToHost);
		

	   /* std::cout << "TESTANDO ( cv::norm(Y) == 0 ) " << std::endl;
		std::cout << "\n Norma CPU = " << cv::norm(Y) << " Norma GPU = " << normY << std::endl;
		system("pause");*/

		if ((normX == 0) || (normY == 0)) {
			break;
		}

		tac("Teste (normX == 0) || (normY == 0) ");
	}		

	/* Copy data from device to host */
	cudaMemcpy(&W.at<float>(0), W_gpu, X.cols*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");
	cudaMemcpy(&P.at<float>(0), P_gpu, X.cols*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (1)");
	cudaMemcpy(&T.at<float>(0), T_gpu, X.rows*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
	
	// return BStar
	cv::Mat bstar = ((W * (P.t() * W).inv()) * (T.t() * T).inv() * T.t() * _Y);
	tac("Total");
	std::cout << "Matriz bstar CUDA [33056][1]: \n";

	for(int i=0;i<20;i++){
		std::cout << bstar.at<float>(i,0) << " " << std::endl;
	}

	system("pause");

	///FREE GPU AQUI
	cudaFree(X_gpu);
	cudaFree(Y_gpu);
	cudaFree(TempX_gpu);
	cudaFree(TempY_gpu);
	cudaFree(T_gpu);
	cudaFree(U_gpu);
	cudaFree(P_gpu);
	cudaFree(W_gpu);
	cudaFree(Q_gpu);
	cudaFree(B_gpu);
	cudaFree(tTemp_gpu);
	cudaFree(uTemp_gpu);
	cudaFree(wTemp_gpu);
	cudaFree(qTemp_gpu);
	cudaFree(tNorm_gpu);
	cudaFree(bTemp_gpu);
	cudaFree(pTemp_gpu);
	cudaFree(hTemp_gpu);
	cudaFree(gTemp_gpu);
	cudaFree(tNew_gpu);
	cudaFree(sub_gpu);
	CHECK(cudaGetLastError(), "Erro ao liberar memoria");

}

void PLSCuda::gpu_FindHighestNormX(cv::Mat X, cv::Mat TempX, double& MaxValX, int& MaxIndexX, float* X_gpu){

	int index3;
	double normX;
	int index;
	cv::Mat X_test;

	X_test.create(X.rows, X.cols, CV_32F);



	for (index3 = 0; index3 < X.cols; index3++) {

		for (int index2 = 0; index2 < X.rows; index2++) {
			TempX.at<float>(index2, 0) = X.at<float>(index2, index3);
		}

	 
			double cv_norm_TempX = cv::norm(TempX);
			round_4(&cv_norm_TempX);

		if (cv_norm_TempX > MaxValX) {
			
			MaxValX = cv_norm_TempX;
			MaxIndexX = index3;
			
			
		}

	}

	
	findhighestXNORMCUDA<<< 1, 1>>>(X_gpu, X.rows, X.cols); CHECK(cudaGetLastError(), "Calcula Maior Norma (1)");
	
	/*cudaMemcpy(&X_test.at<float>(0), X_gpu, X.rows*X.cols*sizeof(float) , cudaMemcpyDeviceToHost);


	std::cout << "MATRIZ X CPU: \n";
	for (int i=0;i<10;i++){
		std::cout << std::endl;
		for (int j=0;j<10;j++)
			std::cout << X.at<float>(i,j) << " ";
	}

	std::cout << "MATRIZ X GPU: \n";
	for (int i=0;i<10;i++){
		std::cout << std::endl;
		for (int j=0;j<10;j++)
			std::cout << X_test.at<float>(i,j) << " ";
	}
	*/
	

	/*std::cout << "TESTANDO A MAIOR NORMA DE X: " << std::endl;
	cudaMemcpyFromSymbol(&index, MaxIndexXDevice, sizeof(index), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&normX, MaxValXDevice, sizeof(normX), 0, cudaMemcpyDeviceToHost);
	
	std::cout << "\n Maior Norma CPU = " << MaxValX << " Index CPU = " << MaxIndexX  << std::endl;
	std::cout << "\n Maior Norma GPU = " << normX << " Index GPU = " << index  << std::endl;
	system("pause");*/

	//cudaMemcpyFromSymbol(&MaxValX, MaxValXDevice, sizeof(MaxValX), 0, cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia Norma para o Host (0)");
	cudaMemcpyFromSymbol(&MaxIndexX, MaxIndexXDevice, sizeof(MaxIndexX), 0, cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia o Index da Coluna para o Host (0)");
	/*std::cout << " Index CPU (Maior Norma de X) = " << MaxIndexX  << std::endl;
	system("pause");*/
	
}

void PLSCuda::gpu_FindHighestNormY(cv::Mat& Y, cv::Mat& TempY, double& MaxValY, int& MaxIndexY, float* Y_gpu){

	
	double normY;
	int index, index3;


	for (index3 = 0; index3 < Y.cols; index3++) {

		for (int index2 = 0; index2 < Y.rows; index2++) {
			TempY.at<float>(index2, 0) = Y.at<float>(index2, index3);
		}
		
		double cv_norm_TempY = cv::norm(TempY);
		round_4(&cv_norm_TempY);

		if (cv_norm_TempY > MaxValY) {
			MaxValY = cv_norm_TempY;
			MaxIndexY = index3;
		}
	}
	


	findhighestYNORMCUDA<<< 1, 1>>>(Y_gpu, Y.rows, Y.cols); CHECK(cudaGetLastError(), "Calcula Maior Norma (2)");
	

	
	/*std::cout << "TESTANDO A MAIOR NORMA DE Y: " << std::endl;
	cudaMemcpyFromSymbol(&normY, MaxValYDevice, sizeof(normY), 0, cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia Norma para o Host (1)");
	cudaMemcpyFromSymbol(&index, MaxIndexYDevice, sizeof(index), 0, cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia o Index da Coluna para o Host (1)");
	std::cout << "\n Maior Norma CPU = " << MaxValY << " Index CPU = " << MaxIndexY  << std::endl;
	std::cout << "\n Maior Norma GPU = " << normY << " Index GPU = " << index  << std::endl;	
	system("pause");*/

	//cudaMemcpyFromSymbol(&MaxValY, MaxValYDevice, sizeof(MaxValY), 0, cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia Norma para o Host (1)");
	cudaMemcpyFromSymbol(&MaxIndexY, MaxIndexYDevice, sizeof(MaxIndexY), 0, cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia o Index da Coluna para o Host (1)");
	
	/*std::cout << " Index CPU (Maior Norma de Y) = " << MaxIndexY  << std::endl;
	system("pause");*/

}


void PLSCuda::gpu_SaveResults_tTemp_uTemp(cv::Mat X, cv::Mat Y, cv::Mat tTemp, cv::Mat uTemp, cv::Mat tTemp_TEST, cv::Mat uTemp_TEST, double MaxIndexX, double MaxIndexY, float* X_gpu, float* Y_gpu, float* tTemp_gpu, float* uTemp_gpu){

	int grid_size;

	/* tTemp.at<float>(index3, 0) = X.at<float>(index3, MaxIndexX);
	uTemp.at<float>(index3, 0) = Y.at<float>(index3, MaxIndexY); */
	
	grid_size = (int)ceil((float)X.rows/BLOCK_SIZE); // ceil arredonda o valor para cima
	copyColumnToVectorCUDA<<< grid_size , dimBlock >>>(X_gpu, tTemp_gpu, X.rows, X.cols, MaxIndexX); CHECK(cudaGetLastError(), "Copia Coluna (2)");
	copyColumnToVectorCUDA<<< grid_size , dimBlock >>>(Y_gpu, uTemp_gpu, Y.rows, Y.cols, MaxIndexY); CHECK(cudaGetLastError(), "Copia Coluna (3)");

	/*std::cout << "TESTANDO (tTemp.at<float>(index3, 0) = X.at<float>(index3, MaxIndexX)) " << std::endl;
	cudaMemcpy(&tTemp_TEST.at<float>(0), tTemp_gpu, X.rows*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i < X.rows;i++){
		if(tTemp.at<float>(i,0)!= tTemp_TEST.at<float>(i,0))
			std::cout << " Original = " << tTemp.at<float>(i,0) << " GPU = " << tTemp_TEST.at<float>(i,0) << std::endl;  
	}*/

	/*for (int i = 0; i < 5;i++){
		std::cout << "GPU = " << tTemp_TEST.at<float>(i,0) << std::endl;
	}
	system("pause");*/

	/*std::cout << "TESTANDO (uTemp.at<float>(index3, 0) = Y.at<float>(index3, MaxIndexY)) " << std::endl;
	cudaMemcpy(&uTemp_TEST.at<float>(0), uTemp_gpu, Y.rows*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i < Y.rows;i++){
		if(uTemp.at<float>(i,0)!= uTemp_TEST.at<float>(i,0))
			std::cout << " Original = " << uTemp.at<float>(i,0) << " GPU = " << uTemp_TEST.at<float>(i,0) << std::endl;  
	}

	for (int i = 0; i < 5;i++){
		std::cout <<" GPU = " << uTemp_TEST.at<float>(i,0) << std::endl;  
	}

	system("pause");*/
	
}

void PLSCuda::gpu_SaveResults_T_U (cv::Mat X, cv::Mat T, cv::Mat U, cv::Mat tTemp, cv::Mat uTemp, cv::Mat T_TEST, cv::Mat U_TEST, int index1, int nMaxIterations, float* T_gpu,float* U_gpu,float* tTemp_gpu,float* uTemp_gpu){

	int grid_size;

	/* T.at<float>(index3, index1) = tTemp.at<float>(index3, 0);
	U.at<float>(index3, index1) = uTemp.at<float>(index3, 0); */

	grid_size = (int)ceil((float)X.rows/BLOCK_SIZE); // ceil arredonda o valor para cima
	copyVectorToColumnCUDA<<< grid_size , dimBlock >>>(T_gpu, tTemp_gpu, X.rows, nMaxIterations, index1); CHECK(cudaGetLastError(), "Copia Vetor para Coluna (0)");
			
	/*std::cout << "TESTANDO (T.at<float>(index3, index1) = tTemp.at<float>(index3, 0)) " << std::endl;
	cudaMemcpy(&T_TEST.at<float>(0), T_gpu, X.rows*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i != X.rows;i++){
		if(T.at<float>(i,index1)!= T_TEST.at<float>(i,index1))
			std::cout << " Original = " << T.at<float>(i,index1) << " GPU = " << T_TEST.at<float>(i,index1) << std::endl;  
	}

	system("pause");*/

	copyVectorToColumnCUDA<<< grid_size , dimBlock >>>(U_gpu, uTemp_gpu, X.rows, nMaxIterations, index1); CHECK(cudaGetLastError(), "Copia Vetor para Coluna (0)");
			
	/*std::cout << "TESTANDO (U.at<float>(index3, index1) = uTemp.at<float>(index3, 0)) " << std::endl;
	cudaMemcpy(&U_TEST.at<float>(0), U_gpu, X.rows*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i != X.rows;i++){
		if(U.at<float>(i,index1)!= U_TEST.at<float>(i,index1))
			std::cout << " Original = " << U.at<float>(i,index1) << " GPU = " << U_TEST.at<float>(i,index1) << std::endl;  
	}

	system("pause");*/

}

void PLSCuda::gpu_SaveResults_P_W (cv::Mat X, cv::Mat P, cv::Mat W, cv::Mat pTemp, cv::Mat wTemp, cv::Mat P_TEST, cv::Mat W_TEST, int index1, int nMaxIterations, float* P_gpu, float* W_gpu, float* pTemp_gpu, float* wTemp_gpu){

	int grid_size;

	/* P.at<float>(index3, index1) = pTemp.at<float>(index3, 0);
	W.at<float>(index3, index1) = wTemp.at<float>(index3, 0); */

	grid_size = (int)ceil((float)X.cols/BLOCK_SIZE); // ceil arredonda o valor para cima
	copyVectorToColumnCUDA<<< grid_size , dimBlock >>>(P_gpu, pTemp_gpu, X.cols, nMaxIterations, index1); CHECK(cudaGetLastError(), "Copia Vetor para Coluna (0)");
			
	/*std::cout << "TESTANDO (P.at<float>(index3, index1) = pTemp.at<float>(index3, 0)) " << std::endl;
	cudaMemcpy(&P_TEST.at<float>(0), P_gpu, X.cols*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i != 40;i++){
		if(P.at<float>(i,index1)!= P_TEST.at<float>(i,index1))
			std::cout << " Original = " << P.at<float>(i,index1) << " GPU = " << P_TEST.at<float>(i,index1) << std::endl;  
	}

	system("pause");*/

	copyVectorToColumnCUDA<<< grid_size , dimBlock >>>(W_gpu, wTemp_gpu, X.cols, nMaxIterations, index1); CHECK(cudaGetLastError(), "Copia Vetor para Coluna (0)");
			
	/*std::cout << "TESTANDO (W.at<float>(index3, index1) = wTemp.at<float>(index3, 0)) " << std::endl;
	cudaMemcpy(&W_TEST.at<float>(0), W_gpu, X.cols*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i != 40;i++){
		if(W.at<float>(i,index1)!= W_TEST.at<float>(i,index1))
			std::cout << " Original = " << W.at<float>(i,index1) << " GPU = " << W_TEST.at<float>(i,index1) << std::endl;  
	}

	system("pause");*/

}

void PLSCuda::gpu_SaveResults_Q(cv::Mat Y, cv::Mat Q, cv::Mat qTemp, cv::Mat Q_TEST, int index1, int nMaxIterations, float* Q_gpu, float* qTemp_gpu){

	int grid_size;

	/* Q.at<float>(index3, index1) = qTemp.at<float>(index3, 0); */

	grid_size = (int)ceil((float)Y.cols/BLOCK_SIZE); // ceil arredonda o valor para cima
	copyVectorToColumnCUDA<<< grid_size , dimBlock >>>(Q_gpu, qTemp_gpu, Y.cols, nMaxIterations, index1); CHECK(cudaGetLastError(), "Copia Vetor para Coluna (0)");
		
	/*std::cout << "TESTANDO (Q.at<float>(index3, index1) = qTemp.at<float>(index3, 0)) " << std::endl;
	cudaMemcpy(&Q_TEST.at<float>(0), Q_gpu, Y.cols*nMaxIterations*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

	for (int i = 0; i != Y.cols;i++){
		if(Q.at<float>(i,index1)!= Q_TEST.at<float>(i,index1))
			std::cout << " Original = " << Q.at<float>(i,index1) << " GPU = " << Q_TEST.at<float>(i,index1) << std::endl;  
	}

	system("pause");*/

}
	
void PLSCuda::gpu_Iterations(cv::Mat X, cv::Mat Y, cv::Mat& tTemp, cv::Mat& uTemp, cv::Mat& wTemp, cv::Mat& qTemp, float *X_gpu, float *Y_gpu, float *tTemp_gpu, float *uTemp_gpu, float *wTemp_gpu, float *qTemp_gpu, float *tNew_gpu, float *sub_gpu, int nMaxOuter){

	double TermCrit = 10e-15;
	double TermCrit2 = 10e-4;
	double normHost;
	double TempVal;

	cv::Mat wTemp_TEST, wTemp_TEST2, tNew_TEST, tNew, qTemp_TEST,qTemp_TEST2, uTemp_TEST, tTemp_TEST, Y_TEST;
	int i,j;

	wTemp_TEST.create(X.cols, 1, CV_32F);
	wTemp_TEST2.create(X.cols, 1, CV_32F);
	tNew_TEST.create(X.rows, 1, CV_32F);
	tNew.create(X.rows, 1, CV_32F);
	qTemp_TEST.create(Y.cols, 1, CV_32F);
	qTemp_TEST2.create(Y.cols, 1, CV_32F);
	uTemp_TEST.create(Y.rows, 1, CV_32F);
	tTemp_TEST.create(X.rows, 1, CV_32F);
	Y_TEST.create(Y.rows,Y.cols,CV_32F);

	
	int grid_size; /* Tamanho do grid a ser utilizado nas operacoes sobre os vetores. */ 
	dim3 dimGrid( 1 , (X.cols + dimBlock.y - 1) / dimBlock.y); /* wTemp = X.t() * uTemp; */
	dim3 dimGrid2( 1 , (X.rows + dimBlock.y - 1) / dimBlock.y); /* tNew = X * wTemp; */
	dim3 dimGrid3( 1 , (Y.cols + dimBlock.y - 1) / dimBlock.y); /* qTemp = Y.t() * tNew; */
	dim3 dimGrid4( 1 , (Y.rows + dimBlock.y - 1) / dimBlock.y); /* uTemp = Y * qTemp; */
	dim3 dimGrid5( 1 , (X.rows + dimBlock.y - 1) / dimBlock.y); /*	tTemp = tNew.clone(); */

	for (int index2 = 0; index2 < nMaxOuter; index2++) {

		/* wTemp = X.t() * uTemp; */
				
		//tic("mulT1");
		mulTransbyNormalCUDA <<< dimGrid, dimBlock >>>(X_gpu, uTemp_gpu, wTemp_gpu, X.rows, X.cols, Y.rows, 1); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");	
		//tac("mulT1");

		wTemp = X.t() * uTemp;

		/*std::cout << "TESTANDO (wTemp = X.t() * uTemp) " << std::endl;
		cudaMemcpy(&wTemp_TEST.at<float>(0), wTemp_gpu,X.cols*uTemp.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (0)");

		for (i = 0; i < 30;i++){
			if(wTemp.at<float>(i,0)!= wTemp_TEST.at<float>(i,0))
			std::cout << "Original = " << wTemp.at<float>(i,0) << " GPU = " << wTemp_TEST.at<float>(i,0) << std::endl;  
			
		}

		system("pause");*/
		

		/* wTemp = wTemp / cv::norm(wTemp); */
		//tic("div1");
		grid_size = (int)ceil((float)X.cols/BLOCK_SIZE); // ceil arredonda o valor para cima 
		normCUDA<<<1,1>>> (wTemp_gpu, X.cols);				
		divVectorbyNormCUDA <<< grid_size, BLOCK_SIZE >>> (wTemp_gpu, X.cols);  CHECK(cudaGetLastError(), "Dividir Vetor por um valor (1)");					
		//tac("div1");

		wTemp = wTemp / cv::norm(wTemp);
		
		/*std::cout << "TESTANDO (wTemp = wTemp / cv::norm(wTemp)) " << std::endl; 
		cudaMemcpy(&wTemp_TEST2.at<float>(0), wTemp_gpu,X.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		for (j =0; j < 40;j++){
			if(wTemp.at<float>(j,0)!= wTemp_TEST2.at<float>(j,0))
			std::cout << "Original = " << wTemp.at<float>(j,0) << " GPU = " << wTemp_TEST2.at<float>(j,0) << std::endl;  
			
		}

		system("pause");*/
		

		/* tNew = X * wTemp; */
		//tic("mul1");
		mulCUDA <<< dimGrid2, dimBlock >>> (X_gpu, wTemp_gpu, tNew_gpu, X.rows, X.cols, X.cols, 1); CHECK(cudaGetLastError(), "Multiplica (1)");
		//tac("mul1");

		tNew = X * wTemp;
		
		/*std::cout << "TESTANDO (tNew = X * wTemp) " << std::endl; 
		cudaMemcpy(&tNew_TEST.at<float>(0), tNew_gpu,X.rows*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<40;i++){
			if(tNew.at<float>(i,0)!= tNew_TEST.at<float>(i,0))
			std::cout << "Original = " << tNew.at<float>(i,0) << " GPU = " << tNew_TEST.at<float>(i,0) << std::endl;  	
		}

		system("pause");*/
		

		/* qTemp = Y.t() * tNew; */
		//tic("mulT2");
		mulTransbyNormalCUDA <<< dimGrid3, dimBlock >>>(Y_gpu, tNew_gpu, qTemp_gpu, Y.rows, Y.cols, X.rows, 1);CHECK(cudaGetLastError(), "Multiplica Transposta (2)");
		//tac("mulT2");
		
		qTemp = Y.t() * tNew;

		/*std::cout << "TESTANDO (qTemp = Y.t() * tNew) " << std::endl;
		cudaMemcpy(&qTemp_TEST.at<float>(0), qTemp_gpu, Y.cols*1*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<40;i++){
			if(qTemp.at<float>(i,0)!= qTemp_TEST.at<float>(i,0))
				std::cout << "Original = " << qTemp.at<float>(i,0) << " GPU = " << qTemp_TEST.at<float>(i,0) << std::endl;  
		}
		system("pause");*/
		

		/* qTemp = qTemp / cv::norm(qTemp); */
		//tic("div2");
		grid_size = (int)ceil((float)Y.cols/BLOCK_SIZE); // ceil arredonda o valor para cima 		
		normCUDA<<<1,1>>> (qTemp_gpu, Y.cols);
		divVectorbyNormCUDA <<< grid_size , BLOCK_SIZE >>> (qTemp_gpu, Y.cols); CHECK(cudaGetLastError(), "Dividir Vetor por um valor (2)");
		//tac("div2");

		qTemp = qTemp / cv::norm(qTemp);
		
		/*std::cout << "TESTANDO (qTemp = qTemp / cv::norm(qTemp)) " << std::endl;
		cudaMemcpy(&qTemp_TEST2.at<float>(0), qTemp_gpu, Y.cols*1*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		for(i=0;i<40;i++){
				if(qTemp.at<float>(i,0)!= qTemp_TEST2.at<float>(i,0))
				std::cout << "Original = " << qTemp.at<float>(i,0) << " GPU = " << qTemp_TEST2.at<float>(i,0) << std::endl;  
			}
		system("pause");*/
		

		/* uTemp = Y * qTemp; */	
		//tic("mul2");
		mulCUDA <<< dimGrid4, dimBlock >>> (Y_gpu, qTemp_gpu, uTemp_gpu, Y.rows, Y.cols, Y.cols, 1); CHECK(cudaGetLastError(), "Multiplica (2)");
		//tac("mul2");

		uTemp = Y * qTemp;

		/*std::cout << "TESTANDO (uTemp = Y * qTemp) " << std::endl;
		cudaMemcpy(&uTemp_TEST.at<float>(0), uTemp_gpu, Y.rows*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		for(i=0;i<40;i++){
				if(uTemp.at<float>(i,0)!= uTemp_TEST.at<float>(i,0))
				std::cout << "Original = " << uTemp.at<float>(i,0) << " GPU = " << uTemp_TEST.at<float>(i,0) << std::endl;  
			}
		system("pause");*/
		


		//tic("final");
		/* TempVal = cv::norm (tTemp - tNew); */
		grid_size = (int)ceil((float)X.rows/BLOCK_SIZE); // ceil arredonda o valor para cima
		subCUDA <<< grid_size , BLOCK_SIZE >>> (tTemp_gpu, tNew_gpu, sub_gpu, X.cols, X.rows*X.cols); CHECK(cudaGetLastError(), "Subtrai dois vetores (1)");
		normCUDA<<<1,1>>> (sub_gpu, X.rows);	
		cudaMemcpyFromSymbol(&normHost, normDevice, sizeof(normHost), 0, cudaMemcpyDeviceToHost);

		

		TempVal = cv::norm (tTemp - tNew);		
		
		//std::cout << "Iteracao do ITERATIONS: " << index2 << std::endl  << " GPU = " << normHost << " TermCrit2 = " << TermCrit2 << std::endl;  		
		//system("pause");

		//std::cout << "Iteracao do ITERATIONS: " << index2 << std::endl << " Original = " << TempVal << " GPU = " << normHost << " TermCrit = " << TermCrit << std::endl;  		
		/*system("pause");*/
		
		if(normHost < TermCrit2){
			//std::cout << "Saiu pela GPU\n" << std::endl;
			break;
		}

		/*if(TempVal < TermCrit){
			std::cout << "Saiu pela CPU\n" << std::endl;
			break;
		}*/

		/* tTemp = tNew.clone(); */
		grid_size = (int)ceil((float)X.rows/BLOCK_SIZE); // ceil arredonda o valor para cima
		copyVectorCUDA<<< dimGrid5 , dimBlock >>>( tNew_gpu, tTemp_gpu, X.rows); CHECK(cudaGetLastError(), "Copia Vetor (1)");
		//tac("final");
		
		tTemp = tNew.clone();
		
		/*std::cout << "TESTANDO (tTemp = tNew.clone()) " << std::endl;
		cudaMemcpy(&tTemp_TEST.at<float>(0), tTemp_gpu, X.rows*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		for(i=0;i<40;i++){
				if(tTemp.at<float>(i,0)!= tTemp_TEST.at<float>(i,0))
				std::cout << "Original = " << tTemp.at<float>(i,0) << " GPU = " << tTemp_TEST.at<float>(i,0) << std::endl;  
			}
		system("pause");*/
		
	}

	/* Copia os resultados da memoria global da GPU para a RAM */
	/*cudaMemcpy(&tTemp.at<float>(0), tTemp_gpu,tTemp.rows*tTemp.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
	cudaMemcpy(&uTemp.at<float>(0), uTemp_gpu,uTemp.rows*uTemp.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (3)");
	cudaMemcpy(&wTemp.at<float>(0), wTemp_gpu,wTemp.rows*wTemp.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (4)");
	cudaMemcpy(&qTemp.at<float>(0), qTemp_gpu,qTemp.rows*qTemp.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (5)");
	*/

}


void PLSCuda::gpu_Deflation(cv::Mat& X,cv::Mat& Y, cv::Mat& tTemp, cv::Mat& uTemp,cv::Mat& qTemp,cv::Mat& tNorm,cv::Mat& bTemp, cv::Mat& pTemp, float *X_gpu, float *Y_gpu, float *tTemp_gpu, float *uTemp_gpu, float *tNorm_gpu, float *qTemp_gpu, float *bTemp_gpu, float *pTemp_gpu, float *hTemp_gpu, float *gTemp_gpu){

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int grid_size; /* Tamanho do grid a ser utilizado nas operacoes sobre os vetores. */ 
	dim3 dimGrid(1 , 1); /* tNorm = tTemp.t() * tTemp; */
	dim3 dimGrid2(1 , 1); /* bTemp = uTemp.t() * tTemp */
	dim3 dimGrid3(1 , (X.cols + dimBlock.y - 1) / dimBlock.y); /* pTemp = X.t() * tTemp; */
	dim3 dimGrid4( (X.cols + dimBlock.x - 1) / dimBlock.x , (X.rows + dimBlock.y - 1) / dimBlock.y); /* hTemp = tTemp * pTemp.t(); */
	dim3 dimGrid5( (X.cols + dimBlock.x - 1) / dimBlock.x, (X.rows + dimBlock.y - 1) / dimBlock.y); /* X = X - hTemp; */
	dim3 dimGrid6( (Y.cols + dimBlock.x - 1) / dimBlock.x , (X.rows + dimBlock.y - 1) / dimBlock.y); /* gTemp = tTemp * qTemp.t(); */	
	dim3 dimGrid7( (Y.cols + dimBlock.x - 1) / dimBlock.x , (X.rows + dimBlock.y - 1) / dimBlock.y);/*gTemp = bTemp.at<float>(0, 0) * gTemp;*/
	dim3 dimGrid8( (Y.cols + dimBlock.x - 1) / dimBlock.x , (Y.rows + dimBlock.y - 1) / dimBlock.y); /* Y = Y - gTemp; */

	cv::Mat hTemp, gTemp;
	cv::Mat tNorm_TEST, bTemp_TEST, pTemp_TEST, hTemp_TEST, X_TEST, Y_TEST, gTemp_TEST;
	cv::Mat tTemp_TEST, qTemp_TEST;

	int i,j;
	tNorm_TEST.create(1, 1, CV_32F);
	bTemp_TEST.create(1, 1, CV_32F);
	pTemp_TEST.create(X.cols,1,CV_32F);
	hTemp_TEST.create(X.rows,X.cols,CV_32F);
	X_TEST.create(X.rows,X.cols,CV_32F);
	Y_TEST.create(Y.rows,Y.cols,CV_32F);
	gTemp_TEST.create(X.rows, Y.cols, CV_32F);

	tTemp_TEST.create(X.rows,1,CV_32F);
	qTemp_TEST.create(Y.cols,1,CV_32F);
	

	// Residual Deflation
		/*tNorm = tTemp.t() * tTemp;
		bTemp = uTemp.t() * tTemp / tNorm.at<float>(0, 0);
		pTemp = X.t() * tTemp / tNorm.at<float>(0, 0);
		X = X - tTemp * pTemp.t();
		Y = Y - bTemp.at<float>(0, 0) * (tTemp * qTemp.t());
		*/
	
		//tNorm = tTemp.t() * tTemp;
		//tic("mulT3");
		mulTransbyNormalCUDA <<< dimGrid, dimBlock >>>(tTemp_gpu, tTemp_gpu, tNorm_gpu, X.rows, 1, X.rows, 1); CHECK(cudaGetLastError(), "Multiplica Tranposta (3)");
		//tac("mulT3");

		tNorm = tTemp.t() * tTemp;
		
		/*std::cout << "TESTANDO (tNorm = tTemp.t() * tTemp) " << std::endl;
		cudaMemcpy(&tNorm_TEST.at<float>(0), tNorm_gpu, tNorm.rows*tNorm.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		std::cout << "Original = " << tNorm.at<float>(0,0) << " GPU = " << tNorm_TEST.at<float>(0,0) << std::endl;  
		
		system("pause");*/

		bTemp = uTemp.t() * tTemp;

		//tic("mulT4");
		mulTransbyNormalCUDA <<< dimGrid2, dimBlock >>>(uTemp_gpu, tTemp_gpu, bTemp_gpu, Y.rows, 1, X.rows, 1); CHECK(cudaGetLastError(), "Multiplica Tranposta (4)");
		//tac("mulT4");

		//bTemp = uTemp.t() * tTemp;
		
		/*std::cout << "TESTANDO (bTemp = uTemp.t() * tTemp) " << std::endl;
		cudaMemcpy(&bTemp_TEST.at<float>(0), bTemp_gpu, 1*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		std::cout << "Original = " << bTemp.at<float>(0,0) << " GPU = " << bTemp_TEST.at<float>(0,0) << std::endl;  
		
		system("pause");*/

		bTemp = bTemp / tNorm.at<float>(0, 0);

		grid_size = (int)ceil((float)1/BLOCK_SIZE); // ceil arredonda o valor para cima
		//tic("div2");
		divVectorbyScalarCUDA <<< grid_size, BLOCK_SIZE >>> (bTemp_gpu, 1, tNorm.at<float>(0,0));  CHECK(cudaGetLastError(), "Dividir Vetor por um valor (3)");
		//tac("div2");

		//bTemp = bTemp / tNorm.at<float>(0, 0);
		
		/*std::cout << "TESTANDO (bTemp = bTemp / tNorm.at<float>(0, 0)) " << std::endl;
		cudaMemcpy(&bTemp_TEST.at<float>(0), bTemp_gpu, 1*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		std::cout << "Original = " << bTemp.at<float>(0,0) << " GPU = " << bTemp_TEST.at<float>(0,0) << std::endl;  
		
		system("pause");*/

		//pTemp = X.t() * tTemp;

		//tic("mulT5");
		mulTransbyNormalCUDA <<< dimGrid3, dimBlock >>>(X_gpu, tTemp_gpu, pTemp_gpu, X.rows, X.cols, X.rows, 1); CHECK(cudaGetLastError(), "Multiplica Tranposta (5)");
		//tac("mulT5");

		pTemp = X.t() * tTemp;
		
		/*std::cout << "TESTANDO (pTemp = X.t() * tTemp) " << std::endl;
		cudaMemcpy(&pTemp_TEST.at<float>(0), pTemp_gpu, X.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<40;i++){
			if(pTemp.at<float>(i,0)!=pTemp_TEST.at<float>(i,0))
				std::cout << "Original = " << pTemp.at<float>(i,0) << " GPU = " << pTemp_TEST.at<float>(i,0) << std::endl;  
		}

		system("pause");*/
		

		//pTemp = pTemp / tNorm.at<float>(0, 0);

		grid_size = (int)ceil((float)X.cols/BLOCK_SIZE); // ceil arredonda o valor para cima
		//tic("div3");
		divVectorbyScalarCUDA <<< grid_size, BLOCK_SIZE >>> (pTemp_gpu, X.cols, tNorm.at<float>(0,0));  CHECK(cudaGetLastError(), "Dividir Vetor por um valor (4)");
		//tac("div3");

		pTemp = pTemp / tNorm.at<float>(0, 0);

		/*std::cout << "TESTANDO (/pTemp = pTemp / tNorm.at<float>(0, 0)) " << std::endl;
		cudaMemcpy(&pTemp_TEST.at<float>(0), pTemp_gpu, X.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<40;i++){
			if(pTemp.at<float>(i,0)!=pTemp_TEST.at<float>(i,0))
				std::cout << "Original = " << pTemp.at<float>(i,0) << " GPU = " << pTemp_TEST.at<float>(i,0) << std::endl;  
		}

		system("pause");*/

		//hTemp = tTemp * pTemp.t();
		//tic("mulT6");
		mulNormalbyTransCUDA <<< dimGrid4, dimBlock >>>(tTemp_gpu, pTemp_gpu, hTemp_gpu, X.rows, 1, X.cols, 1); CHECK(cudaGetLastError(), "Multiplica Tranposta (6)");
		//tac("mulT6");

		hTemp = tTemp * pTemp.t();

		/*std::cout << "TESTANDO (hTemp = tTemp * pTemp.t()) " << std::endl;
		cudaMemcpy(&hTemp_TEST.at<float>(0), hTemp_gpu, X.rows*X.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<30;i++){
			for(j=0;j<30;j++){
				if(hTemp.at<float>(i,j)!=hTemp_TEST.at<float>(i,j))
					std::cout << "Original = " << hTemp.at<float>(i,j) << " GPU = " << hTemp_TEST.at<float>(i,j) << std::endl;  
			}
		}

		system("pause");*/

		// X = X - hTemp;
			
		subCUDA <<<  dimGrid5, dimBlock  >>> (X_gpu, hTemp_gpu, X_gpu, X.cols, X.rows*X.cols); CHECK(cudaGetLastError(), "Subtrai duas matrizes (1)");
		
		X = X - hTemp;
		
		/*std::cout << "TESTANDO (X = X - htemp) " << std::endl;
		cudaMemcpy(&X_TEST.at<float>(0), X_gpu, X.rows*X.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
				
		for(i=0;i<10;i++){
			for(j=0;j<10;j++){
				if(X.at<float>(i,j)!=X_TEST.at<float>(i,j))
					std::cout << "Original = " << X.at<float>(i,j) << " GPU = " << X_TEST.at<float>(i,j) << std::endl;  
			}
		}

		system("pause");*/

		//gTemp = tTemp * qTemp.t();
		
		//tic("mulT7");
		mulNormalbyTransCUDA <<< dimGrid6, dimBlock >>>(tTemp_gpu, qTemp_gpu, gTemp_gpu, X.rows, 1, Y.cols, 1); CHECK(cudaGetLastError(), "Multiplica Tranposta (7)");
		//tac("mulT7");

		gTemp = tTemp * qTemp.t();
		
		/*std::cout << "TESTANDO (gTemp = tTemp * qTemp.t()) " << std::endl;
		cudaMemcpy(&gTemp_TEST.at<float>(0), gTemp_gpu, X.rows * Y.cols * sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
	
		
		for(i=0;i<X.rows;i++){
			for(j=0;j<Y.cols;j++){
				if(gTemp.at<float>(i,j)!=gTemp_TEST.at<float>(i,j))
					std::cout << "Original = " << gTemp.at<float>(i,j) << " GPU = " << gTemp_TEST.at<float>(i,j) << std::endl;  
			}
		}

		system("pause");*/


		//gTemp = bTemp.at<float>(0, 0) * gTemp;

		//mulCUDA <<< dimGrid7, dimBlock >>> (bTemp_gpu, gTemp_gpu, gTemp_gpu, 1, 1, X.rows, Y.cols); CHECK(cudaGetLastError(), "Multiplica (1)");

		multiplyMatrixbyElementCUDA<<< dimGrid7, dimBlock>>>(gTemp_gpu, bTemp_gpu, X.rows, Y.cols); CHECK(cudaGetLastError(), "Multiplica A = A*B[0][0]");

		gTemp = bTemp.at<float>(0, 0) * gTemp;

		/*std::cout << "TESTANDO (gTemp = bTemp.at<float>(0, 0) * gTemp) " << std::endl;	
		cudaMemcpy(&gTemp_TEST.at<float>(0), gTemp_gpu, X.rows*Y.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<10;i++){
			for(j=0;j<10;j++){
				if(gTemp.at<float>(i,j)!=gTemp_TEST.at<float>(i,j))
					std::cout << "Original = " << gTemp.at<float>(i,j) << " GPU = " << gTemp_TEST.at<float>(i,j) << std::endl;  
			}
		}

		system("pause");*/

		//Y = Y - gTemp
		
		subCUDA <<< dimGrid8 , dimBlock >>> (Y_gpu, gTemp_gpu, Y_gpu, Y.cols, Y.rows*Y.cols); CHECK(cudaGetLastError(), "Subtrai dois vetores (3)");
	
		Y = Y - gTemp;

		/*std::cout << "TESTANDO (Y = Y - gTemp) " << std::endl;
		cudaMemcpy(&Y_TEST.at<float>(0), Y_gpu, Y.rows*Y.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
		
		for(i=0;i<Y.rows;i++){
			for(j=0;j<Y.cols;j++)
				if(Y.at<float>(i,j)!=Y_TEST.at<float>(i,j))
					std::cout << "Original = " << Y.at<float>(i,j) << " GPU = " << Y_TEST.at<float>(i,j) << std::endl;  
			
		}

		system("pause");*/

		/* Copia os resultados da memoria global da GPU para a RAM */
		/*cudaMemcpy(&X.at<float>(0), X_gpu, X.rows * X.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (6)");
		cudaMemcpy(&Y.at<float>(0), Y_gpu, Y.rows * Y.cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (7)");
		cudaMemcpy(&bTemp.at<float>(0), bTemp_gpu, 1 * sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (8)");
		cudaMemcpy(&qTemp.at<float>(0), qTemp_gpu, Y.cols * sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (9)");
		cudaMemcpy(&pTemp.at<float>(0), pTemp_gpu, X.cols * sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (10)");
		*/
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA METHODS IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void mulCUDA(float *A, float *B, float *C, int Arows, int Acols, int Brows, int Bcols) {

	float sum = 0.f;

	unsigned int i,j;

	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	int RowinC = blockIdx.y * blockDim.y + threadIdx.y;
	int ColinC = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = 0; i < (BLOCK_SIZE + Acols - 1)/BLOCK_SIZE; i++) {
		for (j = 0; j < BLOCK_SIZE; j++) 
			if ((i*BLOCK_SIZE + j < Acols && Row < Arows) && (i*BLOCK_SIZE + j < Brows && Col < Bcols)) 
				sum += A[Row*Acols + i*BLOCK_SIZE + j] * B[(i*BLOCK_SIZE + j)*Bcols + Col];

	}

	if (Row < Arows && Col < Bcols) 
		C[RowinC*Bcols + ColinC] = sum;

}

__global__ void mulTransbyNormalCUDA(float *A_t, float *B, float *C, int A_trows, int A_tcols, int Brows, int Bcols) {

	
	float sum = 0.f;

	unsigned int i,j;

	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	int RowinC = blockIdx.y * blockDim.y + threadIdx.y;
	int ColinC = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = 0; i < (BLOCK_SIZE + A_trows - 1)/BLOCK_SIZE; i++) {
		for (j = 0; j < BLOCK_SIZE; j++) 
			if ( (i*BLOCK_SIZE + j < A_trows && Row < A_tcols) && (i*BLOCK_SIZE + j < Brows && Col < Bcols)) 
					sum += A_t[(i*BLOCK_SIZE + j)*A_tcols + Row] * B[(i*BLOCK_SIZE + j)*Bcols + Col];
	}
	
	if (Row < A_tcols && Col < Bcols) 
		C[RowinC*Bcols + ColinC] = sum;
		
}

__global__ void mulNormalbyTransCUDA (float *A, float *B_t, float *C, int Arows, int Acols, int B_trows, int B_tcols){

	float sum = 0.f;

	unsigned int i,j;

	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	int RowinC = blockIdx.y * blockDim.y + threadIdx.y;
	int ColinC = blockIdx.x * blockDim.x + threadIdx.x;

	for (i = 0; i < (BLOCK_SIZE + Acols - 1)/BLOCK_SIZE; i++) {
		for (j = 0; j < BLOCK_SIZE; j++) 
			if ((i*BLOCK_SIZE + j < Acols && Row < Arows) && (i*BLOCK_SIZE + j < B_tcols && Col < B_trows)) 
				sum += A[Row*Acols + i*BLOCK_SIZE + j] * B_t[Col*B_tcols + i*BLOCK_SIZE + j];

	}
	
	if (Row < Arows && Col < B_trows) 
		C[RowinC*B_trows + ColinC] = sum;

}

__global__ void copyVectorCUDA(float *A, float *B, int size){

		int Row = blockIdx.y*BLOCK_SIZE+threadIdx.y;
        int Col = blockIdx.x*BLOCK_SIZE+threadIdx.x;
        int index = Row + Col;

        if(index < size){
			B[index] = A[index];
        }

}

__global__ void copyColumnToVectorCUDA(float *A, float *B, int num_rows, int num_cols, int column){

        int Col = blockIdx.x*BLOCK_SIZE+threadIdx.x;
        int index = num_cols*Col + column;

        if(index < num_rows*num_cols){
			B[Col] = A[index];
        }

}

__global__ void copyVectorToColumnCUDA(float *A, float *B, int num_rows, int num_cols, int column){

        int Col = blockIdx.x*BLOCK_SIZE+threadIdx.x;
        int index = num_cols*Col + column;

        if(index < num_rows*num_cols){
			A[index] = B[Col];
        }

}
__global__ void subCUDA(float *A, float *B, float *C, int cols, int size){


		int Row = blockIdx.y*BLOCK_SIZE+threadIdx.y;
        int Col = blockIdx.x*BLOCK_SIZE+threadIdx.x;
        int index = Row*cols + Col;

        if(index < size){
			C[index] = A[index] - B[index];
        }
		

}

__global__ void divVectorbyNormCUDA(float *A, int size){

	// Get our global thread ID
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	// Make sure we do not go out of bounds
	if (i < size)
		A[i] = A[i] / normDevice;

}

__global__ void divVectorbyScalarCUDA(float *A, int size, float value){

	// Get our global thread ID
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	// Make sure we do not go out of bounds
	if (i < size)
		A[i] = A[i] / value;

}

__global__ void mulMatrixbyScalarCUDA(float *A, int rows, int cols, float scalar){


	unsigned int i,j;

	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	
	for (i = 0; i < (BLOCK_SIZE + cols - 1)/BLOCK_SIZE; i++) {
		for (j = 0; j < BLOCK_SIZE; j++) 
			if ((i*BLOCK_SIZE + j < cols && Row < rows)  && (i*BLOCK_SIZE + j < rows && Col < cols) ) 
				A[Row*cols + i*BLOCK_SIZE + j]  = A[Row*cols + i*BLOCK_SIZE + j] * scalar;

	}

}

__global__ void normCUDA(float *A, int size){

	if( 0 == threadIdx.x ) {
		double sum = 0;
		for( int i = 0; i < size; i++ )
			sum += (A[i] * A[i]);
		normDevice = sqrt(sum);
	}

}


__global__ void findhighestXNORMCUDA (float *X, int Xrows, int Xcols){

	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	
	double norm = 0.0;
	

	for (int j = 0; j < Xcols; j++) {
	
		if( 0 == threadIdx.x ) {
	
			double sum = 0.0;
		
			for (int i = 0; i < Xrows; i++) 
				sum += X[i*Xcols+j] * X[i*Xcols+j];

			norm = sqrt(sum);
			((norm*10000 - (int)(norm*10000))) >= 0.5 ? norm = int((norm + 0.0001)*10000)/10000.0 : norm = int(norm*10000)/10000.0;
		

		}

		if (norm > MaxValXDevice){
				MaxValXDevice = norm;
				MaxIndexXDevice = j;	
		}	

	}
			
}


__global__ void findhighestYNORMCUDA (float *Y, int Yrows, int Ycols){

	int i,j;
	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	double norm;

	for (j = 0; j < Ycols; j++) {
		
		if( 0 == threadIdx.x ) {
			
			double sum = 0;

			for (i = 0; i < Yrows; i++) 
				sum += Y[i*Ycols +j] * Y[i*Ycols+j];

			norm = sqrt(sum);
			((norm*10000 - (int)(norm*10000))) >= 0.5 ? norm = int((norm + 0.0001)*10000)/10000.0 : norm = int(norm*10000)/10000.0;
			
		}	
	
		if (norm > MaxValYDevice){
				MaxValYDevice = norm;
				MaxIndexYDevice = j;
		
		}

	}

}


__global__ void multiplyMatrixbyElementCUDA (float *A, float*B, int Arows, int Acols){

	unsigned int i,j;

	int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
	int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	
	auxFirstElement = B[0];

	for (i = 0; i < (BLOCK_SIZE + Acols - 1)/BLOCK_SIZE; i++) {
		for (j = 0; j < BLOCK_SIZE; j++) 
			if ((i*BLOCK_SIZE + j < Acols && Row < Arows)  && (i*BLOCK_SIZE + j < Arows && Col < Acols) ) 
				A[Row*Acols + i*BLOCK_SIZE + j]  = A[Row*Acols + i*BLOCK_SIZE + j] * auxFirstElement;

	}


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA TEST METHODS
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


void PLSCuda::opencvCopy(){

	std::cout << "TESTE DA COPIA " << std::endl;

	cv::Mat A, aux, B;
	int i,j;
	srand( (unsigned)time(NULL) );
	A.create(3, 3, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 1 + (rand()%3);

	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	aux = A.t();
	B = aux.clone();

	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	B.at<float>(0,0) = 100;

	std::cout << "Matriz B NOVA: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	std::cout << "Matriz A NOVA: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

}

void PLSCuda::testDIV(){

	std::cout << "TESTE DA DIVISAO POR ESCALAR " << std::endl;
	
	cv::Mat A, A_TEST;
	float *A_gpu;
	int i,j;
	float value = 2;

	A.create(1000, 1, CV_32F);
	A_TEST.create(A.rows, A.cols, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++){
			A.at<float>(i,j) = 4;
			A_TEST.at<float>(i,j) = A.at<float>(i,j)/value;
		}

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	// Numero de threads no grid 
	int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE); // ceil arredonda o valor para cima 

	divVectorbyScalarCUDA <<< grid_size, BLOCK_SIZE >>> (A_gpu,A.rows,value); CHECK(cudaGetLastError(), "Subtracao de Vetor");
	
	cudaMemcpy(&A.at<float>(0), A_gpu,A.rows*A.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	for(i=0;i<A.rows;i++){
		for(j=0;j<A.cols;j++){
			if(A.at<float>(i,j) != A_TEST.at<float>(i,j))
				std::cout << " A = " << A.at<float>(i,j) << " A_TEST = " << A_TEST.at<float>(i,j) << std::endl;
	
		}
	}
	std::cout << std::endl;

	cudaFree(A_gpu);


}

void PLSCuda::testNORM(){

	std::cout << "TESTE DA NORMA " << std::endl;
	cv::Mat A;
	float *A_gpu;
	int i,j;

	A.create(100, 1, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = i+j;

	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	// Numero de threads no grid 
	int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE); // ceil arredonda o valor para cima 

	//normalizeCUDA<<< grid_size, BLOCK_SIZE >>>(A_gpu,A.rows); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	//sum_normalizeCUDA <<< grid_size, BLOCK_SIZE>>>(A_gpu,norm_gpu,A.rows); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");

	tic("OPERACAO");
	normCUDA<<< 1, 1>>>(A_gpu, A.rows); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	tac("OPERACAO");

	tic("COPIA");
	double resposta;
	cudaMemcpyFromSymbol(&resposta, normDevice, sizeof(resposta), 0, cudaMemcpyDeviceToHost);
	tac("COPIA");

	std::cout << "Norma CPU:" << cv::norm(A) << std::endl;

	std::cout << "Norma GPU: " << resposta << std::endl;

	cudaFree(A_gpu);

}

void PLSCuda::testMULTIPLICACAO(){
	std::cout << "TESTE DA MULTIPLICACAO C = A x B " << std::endl;

	cv::Mat A, B, RESULT_CPU, RESULT_GPU;
	float *A_gpu, *B_gpu, *C_gpu;
	int i,j;

	A.create(4, 2, CV_32F);
	B.create(2,4,CV_32F);
	RESULT_GPU.create(A.rows,B.cols,CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = i+j;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = i+j;

	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &C_gpu, A.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid2( (B.cols + dimBlock.x - 1) / dimBlock.x, (A.rows + dimBlock.y - 1) / dimBlock.y); 
	mulCUDA <<< dimGrid2, dimBlock >>> (A_gpu,B_gpu, C_gpu, A.rows, A.cols, B.rows, B.cols); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	cudaMemcpy(&RESULT_GPU.at<float>(0), C_gpu,A.rows*B.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	std::cout << "Matriz RESULTADO: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << RESULT_GPU.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

}

void PLSCuda::testCOPIA(){

	std::cout << "TESTE DA COPIA B = A " << std::endl;

	cv::Mat A, B;
	float *A_gpu, *B_gpu;
	int i,j;

	A.create(1000, 1, CV_32F);
	B.create(A.rows, A.cols, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 2;


	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (B.cols + dimBlock.x - 1) / dimBlock.x, (B.rows + dimBlock.y - 1) / dimBlock.y); 

	copyVectorCUDA <<< dimGrid, dimBlock >>> (A_gpu,B_gpu, B.rows*B.cols); CHECK(cudaGetLastError(), "Copia Vetor");
	cudaMemcpy(&B.at<float>(0), B_gpu,B.rows*B.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	for(int i=920;i<B.rows;i++){
			std::cout << "Posicao = " << i <<" A = " << A.at<float>(i,0) << " B = " << B.at<float>(i,0) << std::endl;
	}
	std::cout << std::endl;
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);

}




void PLSCuda::testMULTIPLICACAO_TRANSPOSTA(){

	std::cout << "TESTE DA MULTIPLICACAO TRANSPOSTA C = A^t x B " << std::endl;
	cv::Mat A, B, RESULT_GPU,A_t;
	float *A_gpu, *B_gpu, *C_gpu;
	int i,j;

	A.create(3,2, CV_32F);
	B.create(3,2,CV_32F);
	RESULT_GPU.create(A.cols,B.cols,CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			 A.at<float>(i,j) = i+j;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = i+j;

	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	A_t = A.t();

	std::cout << "Matriz A TRANSPOSTA: \n";
	for (i=0;i<A_t.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A_t.cols;j++)
			std::cout << A_t.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &C_gpu, RESULT_GPU.rows * RESULT_GPU.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid2( (B.cols + dimBlock.x - 1) / dimBlock.x, (A.cols + dimBlock.y - 1) / dimBlock.y); 
	mulTransbyNormalCUDA <<< dimGrid2,dimBlock >>> (A_gpu, B_gpu, C_gpu, A.rows, A.cols, B.rows, B.cols); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	
	cudaMemcpy(&RESULT_GPU.at<float>(0), C_gpu, RESULT_GPU.rows*RESULT_GPU.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	std::cout << "Matriz RESULTADO: \n";
	for (i=0;i < RESULT_GPU.rows;i++){
		std::cout << std::endl;
		for(j=0;j < RESULT_GPU.cols;j++)
			std::cout << RESULT_GPU.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

}


void PLSCuda::testMULTIPLICACAO_TRANSPOSTA2(){

	std::cout << "TESTE DA MULTIPLICACAO TRANSPOSTA C = A x B^t " << std::endl;
	cv::Mat A, B, B_t, C, C_TEST;
	float *A_gpu, *B_gpu, *C_gpu;
	int i,j;

	A.create(400,400, CV_32F);
	B.create(400,400,CV_32F);
	C.create(A.rows,B.rows,CV_32F);
	C_TEST.create(A.rows,B.rows,CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = i+j;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = i;

	/*std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	B_t = B.t();

	std::cout << "Matriz B TRANSPOSTA: \n";
	for (i=0;i<B_t.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B_t.cols;j++)
			std::cout << B_t.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	*/

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &C_gpu, A.rows * B.rows * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid2( (C_TEST.cols + dimBlock.x - 1) / dimBlock.x, (C_TEST.rows + dimBlock.y - 1) / dimBlock.y); 
	
	C = A * B.t();

	mulNormalbyTransCUDA <<< dimGrid2, dimBlock >>> (A_gpu, B_gpu, C_gpu, A.rows, A.cols, B.rows, B.cols); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	
	cudaMemcpy(&C_TEST.at<float>(0), C_gpu, C_TEST.rows*C_TEST.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	/*std::cout << "Matriz RESULTADO: \n";
	for (i=0;i<C_TEST.rows;i++){
		std::cout << std::endl;
		for(j=0;j<C_TEST.cols;j++)
			std::cout << C_TEST.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	*/

	for (i=0;i<100;i++){
		for(j=0;j<100;j++)
			if (C.at<float>(i,j)!=C_TEST.at<float>(i,j))
				std::cout  << "CPU = " << C.at<float>(i,j) << " GPU = " << C_TEST.at<float>(i,j) << " " << std::endl;
	}


	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

}


void PLSCuda::testMULTIPLICAPORESCALAR(){

	std::cout << "TESTE DA MULTIPLICACAO POR ESCALAR " << std::endl;
	cv::Mat A, RESULT;
	A.create(20,20, CV_32F);
	RESULT.create(A.rows,A.cols, CV_32F);
	float *A_gpu, scalar;
	int i,j;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.cols + dimBlock.x - 1) / dimBlock.x, (A.rows + dimBlock.y - 1) / dimBlock.y); 
	
	scalar = 2;

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = i+j;

	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	
	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	mulMatrixbyScalarCUDA <<< dimGrid , dimBlock >>>(A_gpu, A.rows, A.cols, scalar); CHECK(cudaGetLastError(), "Multiplica por um Escalar (1)");
	

	cudaMemcpy(&RESULT.at<float>(0), A_gpu, RESULT.rows*RESULT.cols*sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "Matriz RESULTADO: \n";
	for (i=0;i<RESULT.rows;i++){
		std::cout << std::endl;
		for(j=0;j<RESULT.cols;j++)
			std::cout << RESULT.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	
	cudaFree(A_gpu);
}

void PLSCuda::testSUB(){

	std::cout << "TESTE DA SUBTRACAO A = A - B " << std::endl;
	cv::Mat A, B, A_TEST;
	float *A_gpu, *B_gpu;
	int i,j;

	A.create(400, 400, CV_32F);
	B.create(A.rows, A.cols, CV_32F);
	A_TEST.create(A.rows, A.cols, CV_32F);

	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.cols + dimBlock.x - 1) / dimBlock.x, (A.rows + dimBlock.y - 1) / dimBlock.y);
	
	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = i+j;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = 1;

	/*std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	

	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	*/


	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	
	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");	

	subCUDA <<< dimGrid , dimBlock >>> (A_gpu,B_gpu,A_gpu, A.cols, A.rows*A.cols); CHECK(cudaGetLastError(), "Subtracao de Vetor");
	
	A = A - B;

	cudaMemcpy(&A_TEST.at<float>(0), A_gpu,A.rows*A.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	/*std::cout << "Matriz A_CPU: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	std::cout << "Matriz A_GPU: \n";
	for (i=0;i<A_TEST.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A_TEST.cols;j++)
			std::cout << A_TEST.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	*/


	for (i=0;i<10;i++){
		for(j=0;j<10;j++)
			if (A.at<float>(i,j)!=A_TEST.at<float>(i,j))
				std::cout  << "CPU = " << A.at<float>(i,j) << " GPU = " << A_TEST.at<float>(i,j) << " " << std::endl;
	}

	cudaFree(A_gpu);
	cudaFree(B_gpu);

}



void PLSCuda::testCOPIACOLUNA(){

	std::cout << "TESTE DA COPIA COLUNA B [i][0] = A[i][col] " << std::endl;

	cv::Mat A, B;
	float *A_gpu, *B_gpu;
	int i,j;
	int col = 0;

	A.create(30, 1, CV_32F);
	B.create(30, 1, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = i+j;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = -1;

	std::cout << "Matriz A:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int grid_size = (int)ceil((float)B.rows/BLOCK_SIZE);

	copyColumnToVectorCUDA <<< grid_size, dimBlock >>> (A_gpu, B_gpu, A.rows,A.cols, col); CHECK(cudaGetLastError(), "Copia Coluna");
	cudaMemcpy(&B.at<float>(0), B_gpu,B.rows*B.cols*sizeof(float) , cudaMemcpyDeviceToHost);

	std::cout << "\nMatriz B: \n";

	for(int i=0;i<B.rows;i++){
		std::cout << "\n";
		for(int j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);

}


void PLSCuda::testCOPIAVETORCOLUNA(){

	std::cout << "TESTE DA COPIA VETOR PARA COLUNA A[i][col] = b[i][0] " << std::endl;

	cv::Mat A, B;
	float *A_gpu, *B_gpu;
	int i,j;
	int col = 3;

	A.create(30, 30, CV_32F);
	B.create(A.rows, 1, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 0;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = i+j;

	std::cout << "Matriz A:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}

	
	std::cout << "\nMatriz B: \n";

	for(int i=0;i<B.rows;i++){
		std::cout << "\n";
		for(int j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE);

	copyVectorToColumnCUDA <<< grid_size, dimBlock >>> (A_gpu, B_gpu, A.rows,A.cols, col); CHECK(cudaGetLastError(), "Copia Vetor para Coluna");
	cudaMemcpy(&A.at<float>(0), A_gpu,A.rows*A.cols*sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "\nMatriz A DEPOIS:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);

}

void PLSCuda::testSETAELEMENTO(){

	std::cout << "TESTE ONDE SETA O VALOR DE UM ELEMENTO EM UMA MATRIZ A[i][j] = element " << std::endl;

	cv::Mat A, B;
	float *A_gpu, *B_gpu;
	int i,j;
	int rowA, rowB, colA, colB;

	A.create(4, 4, CV_32F);
	B.create(2, 2, CV_32F);
	
	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 0;

	std::cout << "Matriz A:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = 3*i+2*j;

	std::cout << "\nMatriz B:" << std::endl;
	for (i=0;i<B.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}

	
	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");


	rowA = 1;
	colA = 1;
	rowB = 0;
	colB = 1;

	std::cout << "\nA[" << rowA <<"][" << colA << "] = B[" << rowB << "][" << colB << "] "<< std::endl;

	std::cout << "\nB[" << rowB << "][" << colB << "] = " << B.at<float>(rowB,colB) << std::endl;

	cudaMemcpy(&A_gpu[rowA*A.cols + colA], &B_gpu[rowB*B.cols + colB], sizeof(float),cudaMemcpyDeviceToDevice); CHECK(cudaGetLastError(), "Cópia do Device para o Device");
	cudaMemcpy(&A.at<float>(0), A_gpu,A.rows * A.cols * sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "\nMatriz A DEPOIS:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);

}

void PLSCuda::testNORMAMATRIZ(){

	std::cout << "TESTE ONDE SETA EH CALCULADA A NORMA DE UMA MATRIZ X => cv::norm(X) " << std::endl;

	cv::Mat X;
	float *X_gpu;
	int i,j;
	double normX;
	
	X.create(3, 3, CV_32F);
	
	for (i=0;i<X.rows;i++)
		for(j=0;j<X.cols;j++)
			X.at<float>(i,j) = i+j;

	std::cout << "Matriz X:" << std::endl;
	for (i=0;i<X.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<X.cols;j++)
			std::cout << X.at<float>(i,j) << " ";
	}


	
	cudaMalloc((void **) &X_gpu, X.rows * X.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMemcpy(X_gpu, &X.at<float>(0), X.rows * X.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");


	
	normCUDA<<< 1, 1>>>(X_gpu, X.rows*X.cols); CHECK(cudaGetLastError(), "Calcula Norma (1)");
	

	cudaMemcpyFromSymbol(&normX, normDevice, sizeof(normX), 0, cudaMemcpyDeviceToHost);


	std::cout << "\n Norma CPU = " << cv::norm(X) << " Norma GPU = " << normX << std::endl;

	
	cudaFree(X_gpu);
	
}


void PLSCuda::testFINDHIGHESTNORM(){

	std::cout << "TESTE PARA ENCONTRAR A MAIOR NORMA " << std::endl;

	cv::Mat X, TempX;
	float *X_gpu;
	int i,j;
	double normX, MaxValX = 0;
	int index, index3;
	int MaxIndexX = -10;
	float a = 0.01;

	srand( (unsigned)time(NULL) );

	X.create(3, 3, CV_32F);
	
	X.at<float>(0,0) = 1;
	X.at<float>(0,1) = 1.0001;
	X.at<float>(0,2) = 1.0001;
	X.at<float>(1,0) = 1;
	X.at<float>(1,1) = 1.0001;
	X.at<float>(1,2) = 1.0001;
	X.at<float>(2,0) = 1;
	X.at<float>(2,1) = 1.0001;
	X.at<float>(2,2) = 1.00009888;

	std::cout << "Matriz X:" << std::endl;
	for (i=0;i<X.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<X.cols;j++)
			std::cout << X.at<float>(i,j) << " ";
	}
	


	TempX.create(X.rows, 1, X.type());

	for (int index2 = 0; index2 < X.cols; index2++){
		for (int index = 0; index < X.rows; index++) {
			TempX.at<float>(index, 0) = X.at<float>(index, index2);
		}

		double cv_norm_TempX = cv::norm(TempX);
		round_4(&cv_norm_TempX);
		std::cout << "\nIndex = " << index2 << " Norma = " << cv::norm(TempX);
		std::cout << "\nIndex = " << index2 << " Norma ARREDONDADA = " << cv_norm_TempX;

		if (cv_norm_TempX > MaxValX) {
	
			MaxValX = cv_norm_TempX;
			MaxIndexX = index2;
	}

	}
	cudaMalloc((void **) &X_gpu, X.rows * X.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMemcpy(X_gpu, &X.at<float>(0), X.rows * X.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	
	findhighestXNORMCUDA<<< 1, 1>>>(X_gpu, X.rows, X.cols); CHECK(cudaGetLastError(), "Calcula Maior Norma (1)");
	
	/*double n = 3.446649;

	std::cout << "\nNumero ANTES = " << n << std::endl;
	std::cout << "(int)n = " << (int)n << std::endl;
	//std::cout << "((n*10000 - (int)(n*10000))) = " << ((n*10000 - (int)(n*10000))) << std::endl;
	((n*10000 - (int)(n*10000))) >= 0.5 ? n = int((n + 0.0001)*10000)/10000.0 : n = int(n*10000)/10000.0;
	
	std::cout << "Numero DEPOIS = " << n << std::endl;

	system("pause");*/

	cudaMemcpyFromSymbol(&normX, MaxValXDevice, sizeof(normX), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&index3, MaxIndexXDevice, sizeof(index3), 0, cudaMemcpyDeviceToHost);

	std::cout << "\n Maior Norma CPU = " << MaxValX << " Index CPU = " << MaxIndexX  << std::endl;
	std::cout << "\n Maior Norma GPU = " << normX << " Index GPU = " << index3  << std::endl;

	
	cudaFree(X_gpu);

}

void PLSCuda::testMULTIPLICAPORPRIMEIROELEMENTO(){


	std::cout << "TESTE PARA MULTIPLICAR UMA MATRIZ PELO PRIMEIRO ELEMENTO DE OUTRA ( A = A * B[0][0]; ) " << std::endl;

	cv::Mat A, B;
	float *A_gpu, *B_gpu;
	int i,j;

	srand( (unsigned)time(NULL) );

	A.create(20, 1, CV_32F);
	B.create(3, 3, CV_32F);


	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (A.cols + dimBlock.x - 1) / dimBlock.x, (A.rows + dimBlock.y - 1) / dimBlock.y); 

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = rand()%5+1;
			
	
	std::cout << "Matriz A:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	
	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = rand()%5+2;
			
	
	std::cout << "\nMatriz B:" << std::endl;
	for (i=0;i<B.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}

	
	std::cout << "\n PRIMEIRO ELEMENTO DA MATRIZ B: "  << B.at<float>(0,0) << std::endl;

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	

	//multiplyMatrixbyElementCUDA (float *A, float*B, int Arows, int Acols);
	multiplyMatrixbyElementCUDA<<< dimGrid, dimBlock>>>(A_gpu, B_gpu, A.rows, A.cols); CHECK(cudaGetLastError(), "Multiplica A = A*B[0][0]");


	cudaMemcpy(&A.at<float>(0), A_gpu,A.rows * A.cols * sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "\nMatriz A DEPOIS:" << std::endl;
	for (i=0;i<A.rows;i++){
		std::cout <<std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	cudaFree(A_gpu);
	cudaFree(B_gpu);

}

void PLSCuda::test(){

	//opencvCopy();
	//testDIV(); // Teste onde divide todos os elementos de um vetor por um valor de ponto flutuante.
	//testCOPIA(); // Teste onde copia os valores de um vetor para outro.
	//testCOPIACOLUNA(); // Teste o qual copia determinada coluna de uma matriz.
	//testCOPIAVETORCOLUNA(); // Teste o qual copia um vetor para uma determinada coluna de uma matriz.
	//testMULTIPLICACAO(); // Teste de multiplicacao de duas matrizes.
	//testNORM(); // Teste que realiza a normalizacao de um vetor.
	//testSUB(); // Teste que realiza a subtracao de duas matrizes.
	//testMULTIPLICACAO_TRANSPOSTA(); // Teste de multiplicacao de duas matrizes, onde a primeira eh transposta.
	//testMULTIPLICACAO_TRANSPOSTA2(); // Teste de multiplicacao de duas matrizes, onde a segunda eh transposta.
	//testMULTIPLICAPORESCALAR(); // Teste de multiplicacao de uma matriz por um escalar.
	//testSETAELEMENTO(); // Teste o qual seta o valor de um elemento em uma matriz.
	//testNORMAMATRIZ(); // Teste o qual calcula o valor da norma de uma matriz.
	//testFINDHIGHESTNORM(); // Teste o qual calcula a norma de cada coluna de uma matriz, e depois imprime na tela o maior valor encontrado.
	//testMULTIPLICAPORPRIMEIROELEMENTO(); // Teste o qual multiplica uma matriz pelo primeiro elemento de outra matriz.



}