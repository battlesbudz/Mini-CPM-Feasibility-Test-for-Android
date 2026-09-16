#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace liquid_check {
using Json = nlohmann::ordered_json;
inline Json compare(const std::vector<float>& actual, const std::vector<float>& reference) {
 double squared=0, refSquared=0, maxError=0, maxRef=0; size_t nonFinite=0;
 for(size_t i=0;i<actual.size();++i){
  if(!std::isfinite(actual[i])){++nonFinite;continue;}
  double d=double(actual[i])-reference[i];squared+=d*d;refSquared+=double(reference[i])*reference[i];
  maxError=std::max(maxError,std::abs(d));maxRef=std::max(maxRef,std::abs(double(reference[i])));
 }
 double relative=std::sqrt(squared/std::max(refSquared,1e-20));
 return {{"relativeL2",relative},{"maxAbsoluteError",maxError},{"referenceMaxAbsolute",maxRef},
 {"nonFinite",nonFinite},{"pass",nonFinite==0&&relative<=0.02&&maxError<=0.02+0.02*maxRef}};
}
inline std::vector<float> compute(ggml_backend_t backend, ggml_type type, int k,int m,int n,
 const std::vector<uint8_t>& weights,const std::vector<float>& input){
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({2*1024*1024,nullptr,true}),ggml_free);
 if(!ctx)throw std::runtime_error("Test graph allocation failed");
 auto a=ggml_new_tensor_2d(ctx.get(),type,k,m);
 auto b=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,k,n);
 auto c=ggml_mul_mat(ctx.get(),a,b);
 if(!ggml_backend_supports_op(backend,c))throw std::runtime_error("Selected backend does not support test operation");
 auto graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,c);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);
 if(!buffer)throw std::runtime_error("Test tensor allocation failed");
 ggml_backend_tensor_set(a,weights.data(),0,weights.size());
 ggml_backend_tensor_set(b,input.data(),0,input.size()*sizeof(float));
 if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Test compute failed");
 ggml_backend_synchronize(backend);
 std::vector<float> result(m*n);ggml_backend_tensor_get(c,result.data(),0,result.size()*sizeof(float));return result;
}
inline void run(Json& report,const std::function<void(const std::string&)>& progress,
 const std::function<void()>& save){
 report["scope"]="synthetic_matmul_cpu_gpu_correctness";
 report["limitations"]="Synthetic contiguous matrix operations only; a pass does not validate the full model, recurrent state, attention, audio, or speed.";
 report["cases"]=Json::array();report["relativeL2Tolerance"]=0.02;
 report["maxErrorTolerance"]="0.02 + 0.02 * referenceMaxAbsolute";
 report["inputSeed"]=42;report["voiceSelectionIgnored"]=true;
 auto cpuDevice=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
 auto gpuDevice=ggml_backend_dev_by_name("Vulkan0");
 if(!cpuDevice||!gpuDevice)throw std::runtime_error("CPU and Vulkan0 are required");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_dev_init(cpuDevice,nullptr),ggml_backend_free);
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> gpu(ggml_backend_dev_init(gpuDevice,nullptr),ggml_backend_free);
 if(!cpu||!gpu)throw std::runtime_error("Backend initialization failed");
 ggml_backend_cpu_set_n_threads(cpu.get(),4);
 report["cpuBackend"]=ggml_backend_name(cpu.get());report["gpuBackend"]=ggml_backend_name(gpu.get());
 bool all=true;uint32_t seed=42;
 auto value=[&](){seed=1664525u*seed+1013904223u;return float((seed>>8)&0xffff)/32768.f-1.f;};
 for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q4_0})for(int k:{256,2048})for(int n:{1,16,126}){
  const int m=64;std::string name=std::string(ggml_type_name(type))+" K="+std::to_string(k)+" M=64 N="+std::to_string(n);
  report["activeCase"]=name;report["activeBackend"]="prepare";save();progress("Checking "+name);
  std::vector<float> original(k*m),input(k*n),decoded(k*m);
  for(auto& x:original)x=value();for(auto& x:input)x=value();
  std::vector<uint8_t> weights(ggml_row_size(type,k)*m);
  ggml_quantize_chunk(type,original.data(),weights.data(),0,m,k,nullptr);
  if(type==GGML_TYPE_F32)decoded=original;else ggml_get_type_traits(type)->to_float(weights.data(),decoded.data(),k*m);
  std::vector<float> reference(m*n);
  for(int col=0;col<n;++col)for(int row=0;row<m;++row){double sum=0;for(int j=0;j<k;++j)sum+=double(decoded[row*k+j])*input[col*k+j];reference[col*m+row]=float(sum);}
  report["activeBackend"]="CPU";save();auto a=compute(cpu.get(),type,k,m,n,weights,input);
  report["activeBackend"]="Vulkan0";save();auto b=compute(gpu.get(),type,k,m,n,weights,input);
  auto ca=compare(a,reference),cb=compare(b,reference),pair=compare(b,a);
  bool pass=ca["pass"].get<bool>()&&cb["pass"].get<bool>()&&pair["pass"].get<bool>();all&=pass;
  report["cases"].push_back({{"name",name},{"cpuVsScalar",ca},{"gpuVsScalar",cb},{"gpuVsCpu",pair},{"pass",pass},
   {"referenceFirst",std::vector<float>(reference.begin(),reference.begin()+4)},
   {"cpuFirst",std::vector<float>(a.begin(),a.begin()+4)},{"gpuFirst",std::vector<float>(b.begin(),b.begin()+4)}});
  if(!pass&&!report.contains("firstFailure"))report["firstFailure"]=name;
  save();
 }
 report["activeBackend"]="finished";report["correctnessPassed"]=all;
 report["verdict"]=all?"CORRECTNESS_PASS":"CORRECTNESS_FAIL";save();
}
}
