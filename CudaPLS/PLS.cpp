#include "PLS.h"

PLS::PLS(bool debug /*= false*/)
	: _debug(debug){

}

void PLS::tic(std::string part){
	if(this->_debug){
		counter_t counter;
		this->counters[part] = counter;
		this->counters[part].tick();
	}
}

void PLS::tac(std::string part){
	if(this->_debug){
		unsigned long long duration = this->counters[part].tock();
		std::cout << "[DEBUG: " << part << "] = " << duration << std::endl;
	}
}
