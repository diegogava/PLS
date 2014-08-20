#include "util.h"
#include "FeatureExtractionClustering.h"
#include "PLS.h"
#include "PLSCpu.h"
#include "PLSCublas.h"
#include "PLSCuda.cuh"

#define __EXTRACT__ false
#define __DEBUG__ true


//#define __CPU__ "CPU"
//#define __CUBLAS__ "CUBLAS"
#define __CUDA__ "CUDA"

#ifdef __CPU__
#define __METHOD__ __CPU__
#define __INSTANCE__ new PLSCpu(__DEBUG__)
#elif defined __CUBLAS__
#define __METHOD__ __CUBLAS__
#define __INSTANCE__ new PLSCublas(__DEBUG__)
#elif defined __CUDA__
#define __METHOD__ __CUDA__
#define __INSTANCE__ new PLSCuda(__DEBUG__)
#endif

cv::Mat read_file (std::string filename, int lines = -1){

	cv::Mat A;
	std::ifstream file;

	int rows, cols;

	file.open(filename);
	if(!file.is_open()){
		std::cout << "Nao foi possivel abrir o arquivo!" << std::endl;
		exit(2);
	}

	file >> rows >>  cols;
	rows = (lines == -1)?rows:lines;

	A.create(rows, cols, CV_32F);

	for (int i=0; i<A.rows;i++){
		for (int j=0;j<A.cols;j++)
			file >> A.at<float>(i,j);
	}

	file.close();

	return A;
}

void write_file(std::string filename, float *A, int Arows, int Acols){

	std::ofstream file;
	file.open(filename);

	file << Arows << " " << Acols << std::endl;

	for (int i=0; i<Arows;i++){		
		for (int j=0;j<Acols;j++){
			file << A[j*Arows +i] << " ";
		}
		file << std::endl;
	}

	file.close();	
}

int main(int argc, char** argv) {

	/*PLSCuda p(true);
	p.test();
	system("pause");
	return EXIT_SUCCESS;*/
	

	///DEFINIÇÃO MANUAL DOS ARGUMENTOS
	int nGroups_Extaction = 1;
	int n_Klusters = 10;
	int factors = 10; // get factors (#factors in PLS)	
	int n_linhas_test = 100;

	// set counter
	counter_t counter;

	//declare cv::mats 
	cv::Mat feats;
	cv::Mat labels;

	if(__EXTRACT__){

		FeatureExtractionClustering fec("pairs.txt");
		fec.execute(feats, labels, nGroups_Extaction, n_Klusters);

		write_file("feats.txt",&feats.at<float>(0),feats.rows, feats.cols);
		write_file("labels.txt",&labels.at<float>(0),labels.rows, labels.cols);

		exit(0);

	}else{		
		feats = read_file("feats.txt", n_linhas_test);
		labels = read_file("labels.txt", n_linhas_test);		
	}


	PLS* pls = __INSTANCE__;
	pls->run(feats, labels, factors);

	system("pause");
	return EXIT_SUCCESS;


}


