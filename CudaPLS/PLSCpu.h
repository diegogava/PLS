#pragma once

#include "pls.h"

class PLSCpu : public PLS{

public:
	PLSCpu(bool debug);
	void run(cv::Mat feats, cv::Mat labels, const int nfactors);

};