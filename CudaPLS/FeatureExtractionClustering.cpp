#include "FeatureExtractionClustering.h"

#include <omp.h>
#include <fstream>
#include "util.h"
#include <iosfwd>


FeatureExtractionClustering::FeatureExtractionClustering(std::string pair_file_path)
	: pair_file_path(pair_file_path){

}

FeatureExtractionClustering::~FeatureExtractionClustering(void)
{
}

void FeatureExtractionClustering::execute(cv::Mat& feats, cv::Mat& labels, int nGroups, int k_clusters){
	// set num threads
	omp_set_num_threads(4);


	// open files
	std::ifstream stream(this->pair_file_path);
	assert(stream.is_open());


	// read #sets(k) and #matched pairs(n)
	int k, n;
	stream >> k;
	stream >> n;

	k = (nGroups == -1)?k:nGroups;

	// calculate features for each set
	std::cout << "*******************" << std::endl;
	std::cout << "Harvesting Features" << std::endl;

	// read matched

	for (int s = 0; s < k; ++s) {
#pragma omp parallel for schedule(guided)
		for (int l = 0; l < n; ++l) {

			// read file
			std::string subject;
			int im1, im2;
#pragma omp critical
			{
				stream >> subject;
				stream >> im1;
				stream >> im2;
			}

			// calculate feats
			{
				cv::Mat f;
				cbind(f, hog(readimage(subject, im1)));
				cbind(f, gray(readimage(subject, im1)));
				cbind(f, hog(readimage(subject, im2)));
				cbind(f, gray(readimage(subject, im2)));
				ts_rbind(feats, f);
			}
		}
		// read unmatched
#pragma omp parallel for schedule(guided)
		for (int l = 0; l < n; ++l) {
			// read file
			std::string subject1, subject2;
			int im1, im2;
#pragma omp critical
			{
				stream >> subject1;
				stream >> im1;
				stream >> subject2;
				stream >> im2;
			}

			// calculate feats
			{
				cv::Mat f;
				cbind(f, hog(readimage(subject1, im1)));
				cbind(f, gray(readimage(subject1, im1)));
				cbind(f, hog(readimage(subject2, im2)));
				cbind(f, gray(readimage(subject2, im2)));
				ts_rbind(feats, f);
			}
		}
	}
	

	// cluster
	std::cout << "*******************" << std::endl; std::cout << "Clustering [k: " << k_clusters << "]" << std::endl;	
	cv::kmeans(feats, k_clusters, labels,
		cv::TermCriteria(CV_TERMCRIT_EPS, 1000, 0.001),
		10,
		cv::KMEANS_PP_CENTERS);

	labels.convertTo(labels, CV_32F);

}

cv::Mat FeatureExtractionClustering::readimage(const std::string &subject, const int &index) {

	// read pairs
	std::ostringstream imagepath;
	// read first pair
	imagepath.fill('0');
	imagepath << "lfw/" << subject << "/" // folder
		<< subject << "_" << std::setw(4) << index << ".jpg"; // image

	// read image
	cv::Mat image = cv::imread(imagepath.str(), cv::IMREAD_GRAYSCALE);
	if (image.empty()) {
		std::cerr << "image " << imagepath.str() << " not found!" << std::endl;
		exit(EXIT_FAILURE);
	}
	cv::Rect crop;
	crop.width = 128;
	crop.height = 128;
	crop.x = image.cols / 2 - crop.width / 2;
	crop.y = image.rows / 2 - crop.height / 2;
	image = image(crop);


	/*{
	cv::imshow("test", image);
	cv::waitKey();
	}*/


	return image;
}

cv::Mat FeatureExtractionClustering::hog(const cv::Mat image) {

	// calculate descriptor
	cv::Mat descriptor;

	// hog
	std::vector<float> hogDescriptor;
	cv::HOGDescriptor hog(
		cv::Size(128, 128),
		cv::Size(32, 32),
		cv::Size(32, 32),
		cv::Size(32, 32),
		9
		);
	hog.compute(image, hogDescriptor);
	cbind(descriptor, cv::Mat(hogDescriptor, true).reshape(1, 1));

	return descriptor;
}

cv::Mat FeatureExtractionClustering::gray(const cv::Mat image) {

	// calculate descriptor
	cv::Mat descriptor;

	image.convertTo(descriptor, CV_32F);
	descriptor = descriptor.reshape(1, 1);

	return descriptor;
}
