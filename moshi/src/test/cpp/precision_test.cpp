#include "codec_precision.h"
#include "gpu_precision.h"
#include "tensor_trace.h"
#include <ggml-cpu.h>
#ifdef MOSHI_TEST_VULKAN
#include <ggml-vulkan.h>
#endif
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
struct Graph {
    ggml_context* ctx=ggml_init({2*1024*1024,nullptr,true});
    ggml_backend_buffer_t buffer=nullptr;
    ~Graph(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void matrixChecks(ggml_backend_t backend,ggml_backend_t other){
    for(auto shape:{std::array<int,3>{256,1,512},std::array<int,3>{192,1920,32},std::array<int,3>{257,17,1}})
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32}){
        auto [k,m,n]=shape;Graph g;require(g.ctx,"context allocation failed");
        auto a=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F16,k,m),b=ggml_new_tensor_2d(g.ctx,type,k,n);
        auto out=ggml_mul_mat(g.ctx,a,b);auto graph=ggml_new_graph(g.ctx);ggml_build_forward_expand(graph,out);
        g.buffer=ggml_backend_alloc_ctx_tensors(g.ctx,backend);require(g.buffer,"buffer allocation failed");
        std::vector<ggml_fp16_t> ah(k*m),bh(k*n);std::vector<float> af(k*m),bf(k*n);
        // Non-dyadic source values rounded to the actual stored F16 values for the oracle.
        for(int i=0;i<k*m;++i){ah[i]=ggml_fp32_to_fp16(std::sin(i*.193)*.7);af[i]=ggml_fp16_to_fp32(ah[i]);}
        for(int i=0;i<k*n;++i){bh[i]=ggml_fp32_to_fp16(std::cos(i*.127)*.6);bf[i]=ggml_fp16_to_fp32(bh[i]);}
        ggml_backend_tensor_set(a,ah.data(),0,ah.size()*2);
        ggml_backend_tensor_set(b,type==GGML_TYPE_F16?static_cast<void*>(bh.data()):static_cast<void*>(bf.data()),0,ggml_nbytes(b));
        std::vector<float> base(m*n),precise(m*n),again(m*n);
        auto run=[&](std::vector<float>& result){require(moshi_trace_compute(backend,graph)==GGML_STATUS_SUCCESS,"matrix dispatch failed");ggml_backend_tensor_get(out,result.data(),0,result.size()*4);};
        run(base);
        {
            GpuPrecisionScope scope(backend);
            {GpuPrecisionGraph guard(other,graph);require(out->op_params[0]==GGML_PREC_DEFAULT,"policy leaked to another backend");}
            {GpuPrecisionScope nested(other);run(again);require(nested.matmuls==0,"nested scope target ignored");}
            run(precise);require(scope.matmuls==1,"precision not applied to compute path");
            require(out->op_params[0]==GGML_PREC_DEFAULT,"graph precision not restored");
        }
        require(!GpuPrecisionScope::active,"scope leaked");run(again);require(base==again,"baseline changed after FP32 experiment");
        double e=0,ref=0,baseError=0,maxError=0;
        for(int col=0;col<n;++col)for(int row=0;row<m;++row){
            double expected=0;for(int j=0;j<k;++j)expected+=double(af[row*k+j])*bf[col*k+j];
            auto index=col*m+row;require(std::isfinite(precise[index])&&std::isfinite(base[index]),"nonfinite matrix output");
            double d=precise[index]-expected;e+=d*d;ref+=expected*expected;maxError=std::max(maxError,std::abs(d));d=base[index]-expected;baseError+=d*d;
        }
        double relative=std::sqrt(e/ref);
        std::cout<<"MATRIX "<<ggml_backend_name(backend)<<" k="<<k<<" m="<<m<<" n="<<n<<" input="<<ggml_type_name(type)<<" default_nrmse="<<std::sqrt(baseError/ref)<<" fp32_nrmse="<<relative<<" max_error="<<maxError<<std::endl;
        require(relative<1e-5&&maxError<.001,"FP32 matrix differs from double-accumulation oracle");
    }
    std::cout<<"PASS matrix oracle, backend isolation, nested scope and baseline restoration"<<std::endl;
}
int main(int argc,char** argv){
    ggml_backend_t cpu=ggml_backend_cpu_init(),other=ggml_backend_cpu_init(),gpu=nullptr;
    int status=0;
    try{
        require(cpu&&other,"CPU unavailable");ggml_backend_cpu_set_n_threads(cpu,4);ggml_backend_cpu_set_n_threads(other,4);
        matrixChecks(cpu,other);
#ifdef MOSHI_TEST_VULKAN
        require(ggml_backend_vk_get_device_count()>0,"Vulkan unavailable; not skipped");gpu=ggml_backend_vk_init(0);require(gpu,"Vulkan initialization failed");matrixChecks(gpu,cpu);
        require(argc==3,"Expected Mimi model and output directory");std::filesystem::create_directories(argv[2]);
        std::vector<float> input(8*1920);for(size_t i=0;i<4*1920;++i)input[i]=.1*std::sin(2*3.141592653589793*440*i/24000);
        auto result=compareCodecPrecision(cpu,gpu,argv[1],input,argv[2],[](const std::string& s){std::cout<<s<<std::endl;});std::cout<<result.json<<std::endl;
        require(result.error.empty(),"Actual Mimi precision experiment failed");require(result.frames==8&&result.comparedTokens==64&&result.fp32Matmuls>0,"Incomplete actual-codec comparison");
        require(result.audio[0].rms()>.001,"Unusable CPU tone reference");
        for(auto& a:result.audio)require(a.count==8*1920&&!a.collapsed(),"Incomplete or collapsed codec output");
        require(result.audio[5].nrmse()<.05,"FP32 decoder differs excessively from identical CPU tokens");
        require(result.audio[4].nrmse()<.1&&result.audio[6].nrmse()<.1,"FP32 encoder/end-to-end audio differs excessively");
        std::cout<<"PASS real Mimi: 8 frames, seven independent routes, valid tokens, finite audio, FP32 numerical bounds"<<std::endl;
#else
        (void)argc;(void)argv;
#endif
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<std::endl;status=1;}
    if(gpu)ggml_backend_free(gpu);if(other)ggml_backend_free(other);if(cpu)ggml_backend_free(cpu);return status;
}
