#include "PLSCpu.h"

PLSCpu::PLSCpu(bool debug)
	: PLS(debug){

}

void PLSCpu::run(cv::Mat feats, cv::Mat labels, const int nfactors){

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


	cv::Mat T, P, U, Q, W, B, _Y;

	// preserve Y
	_Y = Y.clone();

	//Setting the termination criteria
	int nMaxIterations, nMaxOuter = 1000;
	nMaxIterations = nfactors;
	double TermCrit = 10e-15, TempVal;
	cv::Mat tNorm;
	double MaxValX, MaxValY;
	int MaxIndexX, MaxIndexY;
	cv::Mat TempX, TempY;

	//Matrices for storing the intermediate values.
	cv::Mat tTemp, tNew, uTemp, wTemp, qTemp, pTemp, bTemp;

	//Allocating memory
	T.create(X.rows, nMaxIterations, CV_32F);
	P.create(X.cols, nMaxIterations, CV_32F);
	U.create(Y.rows, nMaxIterations, CV_32F);
	Q.create(Y.cols, nMaxIterations, CV_32F);
	W.create(X.cols, nMaxIterations, CV_32F);
	B.create(nMaxIterations, nMaxIterations, CV_32F);
	tTemp.create(X.rows, 1, CV_32F);
	uTemp.create(Y.rows, 1, CV_32F);
	std::cout << "Versao sequencial" << std::endl;
	
	
	for (int index1 = 0; index1 < nMaxIterations; index1++) {
		std::cout << " Iteracao: " << index1 << std::endl;
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

		// Iteration for Outer Modelling
		for (int index2 = 0; index2 < nMaxOuter; index2++) {
		
			tic("mulT1"); wTemp = X.t() * uTemp;				tac("mulT1");
			tic("div1"); wTemp = wTemp / cv::norm(wTemp);		tac("div1");
			tic("mul1"); tNew = X * wTemp;						tac("mul1");
			tic("mulT2"); qTemp = Y.t() * tNew;					tac("mulT2");
			tic("div2"); qTemp = qTemp / cv::norm(qTemp);		tac("div2");
			tic("mul2"); uTemp = Y * qTemp;						tac("mul2");

			tic("final");
			TempVal = cv::norm(tTemp - tNew);
			//std::cout << index2 << " - " << std::fixed << TempVal << std::endl;

			if (cv::norm(tTemp - tNew) < TermCrit) {
				break;
			}
			tTemp = tNew.clone();
			tac("final");

			//system("pause");

		}

		
		// Residual Deflation
		tic("mulT3");tNorm = tTemp.t() * tTemp; tac("mulT3");
		tic("mulT4"); bTemp = uTemp.t() * tTemp / tNorm.at<float>(0, 0); tac("mulT4");
		tic("mulT5"); pTemp = X.t() * tTemp / tNorm.at<float>(0, 0); tac("mulT5");
		tic("expr1"); X = X - tTemp * pTemp.t(); tac("expr1"); 
		tic("expr2"); Y = Y - bTemp.at<float>(0, 0) * (tTemp * qTemp.t()); tac("expr2");


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

	std::cout << "Matriz bstar CPU [33056][1]: \n";

	for(int i=33030;i<33056;i++){
		std::cout << bstar.at<float>(i,0) << " " << std::endl;
	}
}
