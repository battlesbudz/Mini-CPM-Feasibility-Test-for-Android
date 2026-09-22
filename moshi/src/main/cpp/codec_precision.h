#pragma once
#include <ggml-backend.h>
#include <functional>
#include <string>
#include <vector>
#include "audio_metrics.h"
struct CodecPrecisionResult {
    AudioMetrics audio[7];
    uint64_t mismatches[2]{},comparedTokens=0,fp32Graphs=0,fp32Matmuls=0;
    int frames=0;
    std::string error,json;
};
CodecPrecisionResult compareCodecPrecision(ggml_backend_t cpu,ggml_backend_t gpu,
    const std::string& model,const std::vector<float>& input,const std::string& output,
    const std::function<void(const std::string&)>& event);
