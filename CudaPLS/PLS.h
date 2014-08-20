#pragma once

#include <opencv/cv.h>
#include "util.h"
#include <unordered_map>

class PLS{

public:
	PLS(bool debug = false);

	virtual void run(cv::Mat feats, cv::Mat labels, const int nfactors) = 0;

protected:

	bool _debug;
	void tic(std::string part);
	void tac(std::string part);
	std::unordered_map<std::string, counter_t> counters;

};

