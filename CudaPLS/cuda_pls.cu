#include "cuda_pls.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <time.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL DEFINES
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#define BLOCK_SIZE 16

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// UTIL METHODS IMPLEMENATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CHECK(cudaError_t cudaStatus, std::string prefixMessage){
	if (cudaStatus != cudaSuccess) {
		std::cout << prefixMessage << ": " << cudaGetErrorString(cudaStatus) << std::endl;
		system("pause");
		exit(2);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA METHODS DECLARATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mulCUDA(float *A, float *B, float *C, int Arows, int Acols, int Brows, int Bcols, int Crows, int Ccols);
__global__ void mulTransbyNormalCUDA(float *A_t, float *B, float *C, int A_trows, int A_tcols, int Brows, int Bcols, int Crows, int Ccols);
__global__ void copyVectorCUDA(float *A, float *B, int size);
__global__ void normalizeCUDA(float *A, int size);
__global__ void sum_normalizeCUDA(float *A, double *norm, int size);
__global__ void subCUDA(float *A, float *B, float *C, int size);
__global__ void divCUDA(float *A, double *div, int size);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TEST METHODS IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void opencvCopy(){

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

void testDIV(){

	cv::Mat A;
	float *A_gpu;
	double *div,*div_gpu;
	int i,j;

	A.create(30, 1, CV_32F);
	
	div = (double*) malloc (sizeof(double)*1);
	div[0] = 2; // valor

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 4;
	
	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &div_gpu, 1 * sizeof(double)); CHECK(cudaGetLastError(), "Função Malloc");
	

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(div_gpu, div, 1 * sizeof(double), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	
	// Numero de threads no grid 
    int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE); // ceil arredonda o valor para cima 

	divCUDA <<< grid_size, BLOCK_SIZE >>> (A_gpu,div_gpu,A.rows); CHECK(cudaGetLastError(), "Subtracao de Vetor");
	cudaMemcpy(&A.at<float>(0), A_gpu,A.rows*A.cols*sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "Matriz RESULTADO: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	

	cudaFree(A_gpu);
	cudaFree(div_gpu);


}

void testNORM(){

	cv::Mat A, B;
	float *A_gpu;
	double *norm_cpu, *norm_gpu;
	int i,j;

	A.create(30, 1, CV_32F);
	B.create(30, 1, CV_32F);
	
	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 2;
	
	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	norm_cpu = (double*) malloc (sizeof(double)*1);

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &norm_gpu, 1 * sizeof(double)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	
	// Numero de threads no grid 
    int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE); // ceil arredonda o valor para cima 

	normalizeCUDA<<< grid_size, BLOCK_SIZE >>>(A_gpu,A.rows); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	sum_normalizeCUDA <<< grid_size, BLOCK_SIZE>>>(A_gpu,norm_gpu,A.rows); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
	cudaMemcpy(norm_cpu, norm_gpu,1*sizeof(double) , cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "COPIA PARA O HOST (1)");
	cudaMemcpy(&B.at<float>(0), A_gpu,B.rows*B.cols*sizeof(float) , cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "COPIA PARA O HOST (2)");
	
	std::cout << "Norma: " << norm_cpu[0] << std::endl;

	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	cudaFree(A_gpu);
	cudaFree(norm_gpu);	

}

void testMULTIPLICACAO(){

	cv::Mat A, B, RESULT_CPU, RESULT_GPU;
	float *A_gpu, *B_gpu, *C_gpu;
	int i,j;

	A.create(5, 10, CV_32F);
	B.create(10,1,CV_32F);
	RESULT_GPU.create(5,1,CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 2;
	
	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = 1;

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
	mulCUDA <<< dimGrid2, dimBlock >>> (A_gpu,B_gpu, C_gpu, A.rows, A.cols, B.rows, B.cols, A.rows, B.cols); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
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

void testCOPIA(){

	cv::Mat A, B;
	float *A_gpu, *B_gpu;
	int i,j;

	A.create(30, 1, CV_32F);
	B.create(30, 1, CV_32F);
	
	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 2;
	
	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;


	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	// Numero de threads no grid 
    int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE); // ceil arredonda o valor para cima 

	copyVectorCUDA <<< grid_size, BLOCK_SIZE >>> (A_gpu,B_gpu, A.rows); CHECK(cudaGetLastError(), "Copia Vetor");
	cudaMemcpy(&B.at<float>(0), B_gpu,B.rows*B.cols*sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "Matriz B: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	

	cudaFree(A_gpu);
	cudaFree(B_gpu);

}

void testSUB(){

	cv::Mat A, B, C;
	float *A_gpu, *B_gpu, *C_gpu;
	int i,j;

	A.create(20, 1, CV_32F);
	B.create(20, 1, CV_32F);
	C.create(20, 1, CV_32F);

	for (i=0;i<A.rows;i++)
		for(j=0;j<A.cols;j++)
			A.at<float>(i,j) = 2;
	
	std::cout << "Matriz A: \n";
	for (i=0;i<A.rows;i++){
		std::cout << std::endl;
		for(j=0;j<A.cols;j++)
			std::cout << A.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	for (i=0;i<B.rows;i++)
		for(j=0;j<B.cols;j++)
			B.at<float>(i,j) = 1;
	
	std::cout << "Matriz B: \n";
	for (i=0;i<B.rows;i++){
		std::cout << std::endl;
		for(j=0;j<B.cols;j++)
			std::cout << B.at<float>(i,j) << " ";
	}
	std::cout << std::endl;

	cudaMalloc((void **) &A_gpu, A.rows * A.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &B_gpu, B.rows * B.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");
	cudaMalloc((void **) &C_gpu, C.rows * C.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc");

	cudaMemcpy(A_gpu, &A.at<float>(0), A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");
	cudaMemcpy(B_gpu, &B.at<float>(0), B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device");

	// Numero de threads no grid 
    int grid_size = (int)ceil((float)A.rows/BLOCK_SIZE); // ceil arredonda o valor para cima 

	subCUDA <<< grid_size, BLOCK_SIZE >>> (A_gpu,B_gpu,C_gpu,A.rows); CHECK(cudaGetLastError(), "Subtracao de Vetor");
	cudaMemcpy(&C.at<float>(0), C_gpu,C.rows*C.cols*sizeof(float) , cudaMemcpyDeviceToHost);
	
	std::cout << "Matriz RESULTADO: \n";
	for (i=0;i<C.rows;i++){
		std::cout << std::endl;
		for(j=0;j<C.cols;j++)
			std::cout << C.at<float>(i,j) << " ";
	}
	std::cout << std::endl;
	

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CLASS
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CUDA_PLS::test(){
	 opencvCopy();
	//testDIV(); // Teste onde divide todos os elementos de um vetor por um valor de ponto flutuante.
	//testCOPIA(); // Teste onde copia os valores de um vetor para outro.
	//testMULTIPLICACAO(); // Teste de multiplicacao de duas matrizes.
	//testNORM(); // Teste que realiza a normalizacao de um vetor.
	//testSUB(); // Teste que realiza a subtracao de dois vetores.

}

void CUDA_PLS::learn(const cv::Mat &_X, const cv::Mat &Y, const int factors, int max_factors) {

	float avg = 0;
	float avg_last = 0;
	int n = 0;

	for (int nfactors = 1; nfactors <= max_factors; ++nfactors) {

		if (max_factors == 1) {
			nfactors = factors;
			max_factors = factors;
		}

		cv::Mat _Y;
		_Y = Y.clone();

		// preserve original matrix
		cv::Mat X = _X.clone();

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

		// create labels
		std::vector<float> labels;
		labels.clear();

		std::unordered_set<float> _labels;
		for (int i = 0; i < _Y.rows; ++i)
			_labels.insert(_Y.ptr<float>(0)[i]);
		labels.reserve(_labels.size());
		for (float label : _labels)
			labels.push_back(label);


		// perform one-against-all

		for (int bstar = 0; bstar < labels.size(); ++bstar) {

			counter_t counter;
			counter.tick();

			// reserve memory
			cv::Mat y;
			y.create(X.rows, 1, CV_32F);

			// set all target values to -1
			y = -1.0f;

			// set label target values to +1
			for (int i = 0; i < y.rows; ++i) {
				if (_Y.ptr<float>(0)[i] == labels[bstar])
					y.ptr<float>(0)[i] = 1.0f;
			}

			nipals(X.clone(), y, nfactors);

			{
				std::cout << "PLS " << bstar << "/" << labels.size()
					<< " learned [" << counter.tock() << " seconds]" << std::endl;
			}

			// update ETR
			int seconds = counter.tock();
			n++;
			avg = avg + (seconds / (float) n) - (avg_last / n);
			avg_last = seconds;
			int rem_iters = (labels.size() - bstar - 1) + labels.size()*(max_factors - nfactors);
			long h = seconds/(60*60);
			long m = (seconds%(60*60))/60;
			long s = (seconds%(60*60))%60;
			long etr = avg * rem_iters;
			long etr_h = etr/(60*60);
			long etr_m = (etr%(60*60))/60;
			long etr_s = (etr%(60*60))%60;
			std::cout << "PLS learned [elapsed: " << h << ":" << m << ":" << s
				<< ", ETR: " << etr_h << ":" << etr_m << ":" << etr_s << "]" << std::endl;
		}
	}

	std::cout << "Average PLS learning time: " << avg << " seconds" << std::endl;
}

cv::Mat CUDA_PLS::nipals(cv::Mat X, cv::Mat Y, const int nfactors) {

	cv::Mat T, P, U, Q, W, B, _Y;

	// preserve Y
	_Y = Y.clone();

	//Setting the termination criteria
	int nMaxIterations, nMaxOuter = 1000;
	nMaxIterations = nfactors;
	double TermCrit = 10e-15;
	cv::Mat tNorm;
	double MaxValX, MaxValY;
	int MaxIndexX, MaxIndexY;
	cv::Mat TempX, TempY;

	//Matrices for storing the intermediate values.
	cv::Mat tTemp, tNew, uTemp, wTemp, qTemp, pTemp, bTemp;

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

	for (int index1 = 0; index1 < nMaxIterations; index1++) {

		//Finding the column having the highest norm
		MaxValX = 0;
		MaxValY = 0;
		MaxIndexX = -10;
		MaxIndexY = -10;
		TempX.create(X.rows, 1, X.type());
		TempY.create(Y.rows, 1, Y.type());

		for (int index3 = 0; index3 < X.cols; index3++) {
			for (int index2 = 0; index2 < X.rows; index2++) {
				TempX.at<float>(index2, 0) = X.at<float>(index2, index3);
			}
			if (cv::norm(TempX) > MaxValX) {
				MaxValX = cv::norm(TempX);
				MaxIndexX = index3;
			}
		}
		for (int index3 = 0; index3 < Y.cols; index3++) {
			for (int index2 = 0; index2 < Y.rows; index2++) {
				TempY.at<float>(index2, 0) = Y.at<float>(index2, index3);
			}
			if (cv::norm(TempY) > MaxValY) {
				MaxValY = cv::norm(TempY);
				MaxIndexY = index3;
			}
		}

		for (int index3 = 0; index3 < X.rows; index3++) {
			tTemp.at<float>(index3, 0) = X.at<float>(index3, MaxIndexX);
			uTemp.at<float>(index3, 0) = Y.at<float>(index3, MaxIndexY);
		}


		gpu_iterations(X, Y, &tTemp, &uTemp, &wTemp, &qTemp, nMaxOuter, TermCrit);


		// Residual Deflation
		tNorm = tTemp.t() * tTemp;
		bTemp = uTemp.t() * tTemp / tNorm.at<float>(0, 0);
		pTemp = X.t() * tTemp / tNorm.at<float>(0, 0);
		X = X - tTemp * pTemp.t();
		Y = Y - bTemp.at<float>(0, 0) * (tTemp * qTemp.t());


		// Saving Results to Outputs.
		for (int index3 = 0; index3 != X.rows; index3++) {
			T.at<float>(index3, index1) = tTemp.at<float>(index3, 0);
			U.at<float>(index3, index1) = uTemp.at<float>(index3, 0);
		}
		for (int index3 = 0; index3 != X.cols; index3++) {
			P.at<float>(index3, index1) = pTemp.at<float>(index3, 0);
			W.at<float>(index3, index1) = wTemp.at<float>(index3, 0);
		}

		for (int index3 = 0; index3 != qTemp.rows; index3++) {
			Q.at<float>(index3, index1) = qTemp.at<float>(index3, 0);
		}
		B.at<float>(index1, index1) = bTemp.at<float>(0, 0);

		// Checking the residue
		if ((cv::norm(X) == 0) || (cv::norm(Y) == 0)) {
			break;
		}

	}		


	// return BStar
	cv::Mat bstar = ((W * (P.t() * W).inv()) * (T.t() * T).inv() * T.t() * _Y);

	return bstar;
}

void CUDA_PLS::gpu_iterations(cv::Mat X, cv::Mat Y, cv::Mat *tTemp, cv::Mat *uTemp, cv::Mat *wTemp, cv::Mat *qTemp, int nMaxOuter, double TermCrit){
//void CUDA_PLS::gpu_iterations(cv::Mat X, cv::Mat Y, cv::Mat tTemp, cv::Mat uTemp, cv::Mat wTemp, cv::Mat qTemp, int nMaxOuter, double TermCrit){


	//Matrices for storing the intermediate values.
	//cv::Mat tNew, pTemp, bTemp;
	//cv::Mat tNew;
	cv:: Mat aux, X_t, Y_t;
	double *norm_gpu, *TempVal;
	
	float *X_gpu, *Y_gpu, *tTemp_gpu, *uTemp_gpu, *wTemp_gpu, *qTemp_gpu, *tNew_gpu, *wTemp_norm_gpu, *qTemp_norm_gpu, *sub_gpu;
	float *X_t_gpu, *Y_t_gpu;

	aux = X.t();
	X_t = aux.clone();
	aux = Y.t();
	Y_t = aux.clone();


	/* alocando memoria RAM */
	TempVal = (double*) malloc (sizeof(double)*1); 

	/* alocando memoria global da GPU */
	cudaMalloc((void **) &X_gpu, X.rows * X.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (1)");
	cudaMalloc((void **) &Y_gpu, Y.rows * Y.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (2)");
	cudaMalloc((void **) &X_t_gpu, X_t.rows * X_t.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (1)");
	cudaMalloc((void **) &Y_t_gpu, Y_t.rows * Y_t.cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (2)");
	cudaMalloc((void **) &tTemp_gpu, tTemp->rows * tTemp->cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (3)");
	cudaMalloc((void **) &uTemp_gpu, uTemp->rows * uTemp->cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (4)");

	cudaMalloc((void **) &wTemp_gpu, wTemp->rows * wTemp->cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (5)");
	cudaMalloc((void **) &qTemp_gpu, qTemp->rows * qTemp->cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (6)");
	cudaMalloc((void **) &wTemp_norm_gpu, wTemp->rows * wTemp->cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (7)");
	cudaMalloc((void **) &qTemp_norm_gpu, qTemp->rows * qTemp->cols * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (8)");
	cudaMalloc((void **) &tNew_gpu, X.rows * 1 * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (9)");
	cudaMalloc((void **) &sub_gpu, X.rows * 1 * sizeof(float)); CHECK(cudaGetLastError(), "Função Malloc (10)");

	cudaMalloc((void **) &norm_gpu, 1 * sizeof(double)); CHECK(cudaGetLastError(), "Função Malloc(11)");

	/* Copia dados da RAM para a memoria global da GPU */
	cudaMemcpy(X_gpu, &X.at<float>(0), X.rows * X.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device (1)");
	cudaMemcpy(Y_gpu, &Y.at<float>(0), Y.rows * Y.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device (2)");
	cudaMemcpy(X_t_gpu, &X_t.at<float>(0), X_t.rows * X_t.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device (1)");
	cudaMemcpy(Y_t_gpu, &Y_t.at<float>(0), Y_t.rows * Y_t.cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device (2)");
	cudaMemcpy(tTemp_gpu, &tTemp->at<float>(0), tTemp->rows * tTemp->cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device(3)");
	cudaMemcpy(uTemp_gpu, &uTemp->at<float>(0), uTemp->rows * uTemp->cols * sizeof(float), cudaMemcpyHostToDevice); CHECK(cudaGetLastError(), "Cópia para o Device(4)");

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 dimBlock2(1, BLOCK_SIZE);
	int grid_size; /* Tamanho do grid a ser utilizado nas operacoes sobre os vetores. */ 
	dim3 dimGrid( 1 , (X.cols + dimBlock.y - 1) / dimBlock.y); /* wTemp = X.t() * uTemp; */
	dim3 dimGrid2( 1 , (X.rows + dimBlock.y - 1) / dimBlock.y); /* tNew = X * wTemp; */
	dim3 dimGrid3( 1 , (Y.cols + dimBlock.y - 1) / dimBlock.y); /* qTemp = Y.t() * tNew; */
	dim3 dimGrid4( 1 , (Y.rows + dimBlock.y - 1) / dimBlock.y); /* uTemp = Y * qTemp; */

	for (int index2 = 0; index2 < nMaxOuter; index2++) {

		std::cout << "Interação: " << index2 << "/" << nMaxOuter << std::endl;

		/* wTemp = X.t() * uTemp; */
		mulCUDA <<< dimGrid, dimBlock >>>(X_t_gpu, uTemp_gpu, wTemp_gpu, X_t.rows, X_t.cols, uTemp->rows, uTemp->cols, X_t.rows, uTemp->cols); CHECK(cudaGetLastError(), "Multiplica Tranposta (1)");
		
		/* wTemp = wTemp / cv::norm(wTemp); */
		grid_size = (int)ceil((float)X.cols/BLOCK_SIZE); // ceil arredonda o valor para cima 
		copyVectorCUDA <<< grid_size , BLOCK_SIZE >>> (wTemp_gpu, wTemp_norm_gpu, X.cols); CHECK(cudaGetLastError(), "Copia Vetor (1)");
		normalizeCUDA<<< grid_size, BLOCK_SIZE >>>(wTemp_norm_gpu,X.cols); CHECK(cudaGetLastError(), "Normaliza Vetor (1)");
		sum_normalizeCUDA <<< grid_size, BLOCK_SIZE >>>(wTemp_norm_gpu,norm_gpu,X.cols); CHECK(cudaGetLastError(), "Soma para obter a Norma (1)");
		divCUDA <<< grid_size, BLOCK_SIZE >>> (wTemp_gpu, norm_gpu, X.cols);  CHECK(cudaGetLastError(), "Dividir Vetor por um valor (1)");

		/* tNew = X * wTemp; */
		mulCUDA <<< dimGrid2, dimBlock >>> (X_gpu,wTemp_gpu, tNew_gpu, X.rows, X.cols, X.cols, 1, X.rows, 1); CHECK(cudaGetLastError(), "Multiplica (1)");

		/* qTemp = Y.t() * tNew; */
		mulCUDA <<< dimGrid3, dimBlock >>>(Y_t_gpu, tNew_gpu, qTemp_gpu, Y_t.rows, Y_t.cols, X.rows, 1, Y_t.rows, 1);CHECK(cudaGetLastError(), "Multiplica Transposta (2)");
		
		
		/* qTemp = qTemp / cv::norm(qTemp); */
		grid_size = (int)ceil((float)Y.cols/BLOCK_SIZE); // ceil arredonda o valor para cima 
		copyVectorCUDA <<< grid_size , BLOCK_SIZE >>> (qTemp_gpu, qTemp_norm_gpu, Y.cols);CHECK(cudaGetLastError(), "Copia Vetor (2)");
		normalizeCUDA<<< grid_size , BLOCK_SIZE >>>(qTemp_norm_gpu,Y.cols); CHECK(cudaGetLastError(), "Normaliza Vetor (2)");
		sum_normalizeCUDA <<< grid_size , BLOCK_SIZE >>>(qTemp_norm_gpu,norm_gpu,Y.cols); CHECK(cudaGetLastError(), "Soma para obter a Norma (2)");
		divCUDA <<< grid_size , BLOCK_SIZE >>> (qTemp_gpu, norm_gpu, Y.cols); CHECK(cudaGetLastError(), "Dividir Vetor por um valor (2)");

		
		/* uTemp = Y * qTemp; */	
		mulCUDA <<< dimGrid4, dimBlock >>> (Y_gpu,qTemp_gpu, uTemp_gpu, Y.rows, Y.cols, Y.cols, 1, Y.rows, 1); CHECK(cudaGetLastError(), "Multiplica (2)");
		

		/* TempVal = cv::norm (tTemp - tNew); */
		grid_size = (int)ceil((float)X.rows/BLOCK_SIZE); // ceil arredonda o valor para cima
		subCUDA <<< grid_size , BLOCK_SIZE >>> (tTemp_gpu, tNew_gpu, sub_gpu, X.rows); CHECK(cudaGetLastError(), "Subtrai dois vetores (1)");
		normalizeCUDA<<< grid_size , BLOCK_SIZE >>>(sub_gpu,X.rows); CHECK(cudaGetLastError(), "Normaliza Vetor (3)");
		sum_normalizeCUDA <<< grid_size , BLOCK_SIZE >>>(sub_gpu,norm_gpu,X.rows); CHECK(cudaGetLastError(), "Soma para obter a Norma (3)");
		
		cudaMemcpy(TempVal,norm_gpu, 1*sizeof(double) , cudaMemcpyDeviceToHost); CHECK(cudaGetLastError(), "Copia dados do device para o HOST (1)");
			
		if(TempVal[0] < TermCrit){
			break;
		}

		/* tTemp = tNew.clone(); */
		grid_size = (int)ceil((float)X.rows/BLOCK_SIZE); // ceil arredonda o valor para cima
		copyVectorCUDA<<< grid_size , BLOCK_SIZE >>>(tTemp_gpu, tNew_gpu, X.rows); CHECK(cudaGetLastError(), "Copia Vetor (3)");

	}

	/* Copia os resultados da memoria global da GPU para a RAM */
	cudaMemcpy(&tTemp->at<float>(0), tTemp_gpu,tTemp->rows*tTemp->cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (2)");
	cudaMemcpy(&uTemp->at<float>(0), uTemp_gpu,uTemp->rows*uTemp->cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (3)");
	cudaMemcpy(&wTemp->at<float>(0), wTemp_gpu,wTemp->rows*wTemp->cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (4)");
	cudaMemcpy(&qTemp->at<float>(0), qTemp_gpu,qTemp->rows*qTemp->cols*sizeof(float) , cudaMemcpyDeviceToHost);CHECK(cudaGetLastError(), "Copia dados do device para o HOST (5)");

	/* Release device memory */
	cudaFree(X_gpu);
	cudaFree(Y_gpu);
	cudaFree(tTemp_gpu);
	cudaFree(uTemp_gpu);
	cudaFree(wTemp_gpu);
	cudaFree(qTemp_gpu);
	cudaFree(wTemp_norm_gpu);
	cudaFree(qTemp_norm_gpu);
	cudaFree(tNew_gpu);
	cudaFree(sub_gpu);
	cudaFree(norm_gpu);

	/* Release host memory */
	free(TempVal);
	
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA METHODS IMPLEMENTATION
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void mulCUDA(float *A, float *B, float *C, int Arows, int Acols, int Brows, int Bcols, int Crows, int Ccols) {

	// Each thread computes one element of C
	// by accumulating results into sum

	/*float sum = 0.f;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row > Arows || col > Bcols) 
	return;

	for (int i = 0; i < Acols; i++)
	sum += A[row * Acols + i] * B[i * Bcols + col];

	C[row * Crows + col] = sum;
	*/

	float sum = 0.f;

	unsigned int i,j;

    int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y; 
    int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	int RowinC = blockIdx.y * blockDim.y + threadIdx.y;
	int ColinC = blockIdx.x * blockDim.x + threadIdx.x;

	/*A cada iteracao do primeiro laco equivale a um bloco. */
    for (i = 0; i < (BLOCK_SIZE + Acols - 1)/BLOCK_SIZE; i++) {
        for (j = 0; j < BLOCK_SIZE; j++) 
            if ((i*BLOCK_SIZE + j < Acols && Row < Arows) && (i*BLOCK_SIZE + j < Brows && Col < Bcols)) /* Verifica se nao ultrapassou os blocos, e tambem se nao ultrapassou os indices da matriz C. */
                sum += A[Row*Acols + i*BLOCK_SIZE + j] * B[(i*BLOCK_SIZE + j)*Bcols + Col];

    }

    if (Row < Crows && Col < Ccols) 
		C[RowinC*Ccols + ColinC] = sum;

}

__global__ void mulTransbyNormalCUDA(float *A_t, float *B, float *C, int A_trows, int A_tcols, int Brows, int Bcols, int Crows, int Ccols) {

	// Each thread computes one element of C
	// by accumulating results into Cvalue
	
	float sum = 0.f;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row > A_trows || col > Bcols) 
	return;

	for (int i = 0; i < A_tcols; i++)
	sum += A_t[row * A_tcols + i] * B[i * Bcols + col];

	C[row * Crows + col] = sum;


	
}
/* Copies the data vector A to vector B */
__global__ void copyVectorCUDA(float *A, float *B, int size){

	// Get our global thread ID
    int i = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (i < size)
        B[i] = A[i];

}

__global__ void normalizeCUDA(float *A, int size){

	// Get our global thread ID
    int i = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (i < size)
        A[i] = A[i]*A[i];
		
	
}

__global__ void sum_normalizeCUDA(float *A, double *norm, int size){

	norm[0] = 0.0;

	for (int i = 0; i < size; i++){
		norm[0]+=A[i];
	}

	norm[0] = sqrt(norm[0]);

}


__global__ void subCUDA(float *A, float *B, float *C, int size){

	// Get our global thread ID
    int i = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (i < size)
        C[i] = A[i] - B[i];
	
}

__global__ void divCUDA(float *A, double *div, int size){

	// Get our global thread ID
    int i = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (i < size)
        A[i] = A[i] / div[0];
	
}

