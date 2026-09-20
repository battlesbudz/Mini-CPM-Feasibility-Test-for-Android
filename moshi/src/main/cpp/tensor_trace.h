#pragma once
#include <ggml-backend.h>
#include <functional>
#include <memory>
#include <string>
// Thread-local, opt-in diagnostics. Normal benchmarks take the unchanged graph path.
class TensorTrace {
public:
    TensorTrace(const std::string& directory, ggml_backend_t tested, ggml_backend_t reference,
                std::function<void(const std::string&)> progress);
    ~TensorTrace();
    TensorTrace(const TensorTrace&)=delete;
    TensorTrace& operator=(const TensorTrace&)=delete;
    void stage(const std::string& name);
    std::string summary() const;
    struct Impl;
private:
    std::unique_ptr<Impl> impl;
};
ggml_status moshi_trace_compute(ggml_backend_t backend, ggml_cgraph* graph);
void moshi_trace_tensor_set(ggml_tensor* tensor,const void* data,size_t offset,size_t size);
