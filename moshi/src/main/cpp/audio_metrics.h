#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

struct AudioMetrics {
    uint64_t count=0, clipped=0;
    double energy=0, referenceEnergy=0, squaredError=0, cross=0, peak=0;
    void add(float sample, float reference) {
        if(!std::isfinite(sample)||!std::isfinite(reference))
            throw std::runtime_error("Non-finite audio in codec comparison");
        ++count; energy+=double(sample)*sample; referenceEnergy+=double(reference)*reference;
        double difference=double(sample)-reference; squaredError+=difference*difference;
        cross+=double(sample)*reference; peak=std::max(peak,std::abs(double(sample)));
        if(std::abs(sample)>1) ++clipped;
    }
    double rms() const {return count?std::sqrt(energy/count):0;}
    double referenceRms() const {return count?std::sqrt(referenceEnergy/count):0;}
    double nrmse() const {return referenceEnergy>1e-20?std::sqrt(squaredError/referenceEnergy):0;}
    double levelRatio() const {return referenceEnergy>1e-20?std::sqrt(energy/referenceEnergy):0;}
    double correlation() const {return energy*referenceEnergy>1e-20?cross/std::sqrt(energy*referenceEnergy):0;}
    bool collapsed() const {return referenceRms()>=0.01 && levelRatio()<0.01;}
};

inline bool codecLevelFailure(double inputRms, double outputRms) {
    return inputRms>=0.01 && outputRms<inputRms*0.01;
}
