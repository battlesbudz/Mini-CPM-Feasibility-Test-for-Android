#include "correctness.h"
#include <cassert>
#include <limits>
int main(){
 using liquid_check::compare;
 assert(compare({1,2,-3},{1,2,-3})["pass"].get<bool>());
 assert(!compare({0,0,0},{1,2,-3})["pass"].get<bool>());
 assert(!compare({1,2,30},{1,2,-3})["pass"].get<bool>());
 assert(!compare({std::numeric_limits<float>::quiet_NaN()},{1})["pass"].get<bool>());
 assert(!compare({std::numeric_limits<float>::infinity()},{1})["pass"].get<bool>());
 assert(compare({0},{0})["pass"].get<bool>());
}
