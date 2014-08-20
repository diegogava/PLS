#pragma once
#include <string>
#include <opencv\cv.h>

class FeatureExtractionClustering
{
public:
	FeatureExtractionClustering(std::string pair_file_path);
	~FeatureExtractionClustering(void);

	void execute(cv::Mat& feats, cv::Mat& labels, int nGroups = -1, int k_clusters = 10);

private:
	cv::Mat readimage(const std::string &subject, const int &index);
	cv::Mat hog(const cv::Mat image);
	cv::Mat gray(const cv::Mat image);


private:
	std::string pair_file_path;

};

