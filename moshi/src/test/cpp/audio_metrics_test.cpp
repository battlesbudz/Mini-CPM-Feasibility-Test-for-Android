#include "audio_metrics.h"
#include <limits>
#include <cstdio>
#define CHECK(condition) do {if(!(condition)){fprintf(stderr,"Failed line %d: %s\n",__LINE__,#condition);return 1;}} while(false)
int main() {
    AudioMetrics same,collapsed,inverted,quiet;
    for(int i=0;i<1000;++i) {
        float x=.2f*std::sin(i*.1f);
        same.add(x,x);collapsed.add(x*.0005f,x);inverted.add(-x,x);quiet.add(0,0);
    }
    CHECK(same.nrmse()==0 && std::abs(same.levelRatio()-1)<1e-12 && !same.collapsed());
    CHECK(collapsed.collapsed() && collapsed.nrmse()>.999 && collapsed.levelRatio()<.001);
    CHECK(inverted.nrmse()>1.99 && inverted.correlation()<-.99 && !inverted.collapsed());
    CHECK(!quiet.collapsed() && quiet.nrmse()==0);
    CHECK(codecLevelFailure(.20036,.0001168)); // Device regression: Vulkan near-silence.
    CHECK(!codecLevelFailure(.20036,.18209)); // Device CPU reference.
    CHECK(!codecLevelFailure(.0001,0)); // Quiet input is inconclusive, not a failed codec.
    bool threw=false;try{same.add(std::numeric_limits<float>::quiet_NaN(),0);}catch(const std::runtime_error&){threw=true;}
    CHECK(threw);
    puts("PASS: equality, collapsed output, phase inversion, quiet input and nonfinite samples");
}
