// Include the implementation to exercise its private stride sampler and comparator.
#include "tensor_trace.cpp"
#include <ggml-cpu.h>
#include <limits>
#include <iostream>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){
    auto backend=ggml_backend_cpu_init();
    auto ctx=ggml_init({1024*1024,nullptr,true});
    auto base=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,3);
    auto transpose=ggml_transpose(ctx,base);
    auto half=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,3);
    auto bf=ggml_new_tensor_1d(ctx,GGML_TYPE_BF16,3);
    auto ints=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,3);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    try{
        float values[12];for(int i=0;i<12;++i)values[i]=i+1;
        ggml_backend_tensor_set(base,values,0,sizeof(values));
        require(readTensor(transpose).values==std::vector<double>({1,5,9,2,6,10,3,7,11,4,8,12}),"strided transpose sampling");
        float f[3]={0,-1,0.5f};ggml_fp16_t h[3];ggml_bf16_t b[3];int32_t k[3]={0,-7,2047};
        for(int i=0;i<3;++i){h[i]=ggml_fp32_to_fp16(f[i]);b[i]=ggml_fp32_to_bf16(f[i]);}
        ggml_backend_tensor_set(half,h,0,sizeof(h));ggml_backend_tensor_set(bf,b,0,sizeof(b));ggml_backend_tensor_set(ints,k,0,sizeof(k));
        require(readTensor(half).values==std::vector<double>({0,-1,0.5}),"F16 decode");
        require(readTensor(bf).values==readTensor(half).values,"BF16 decode");
        require(readTensor(ints).values==std::vector<double>({0,-7,2047}),"I32 decode");
        Snapshot a,bad;a.values={1,2,3};bad.values={1,2,4};
        require(!Difference(a,a,false).mismatch,"identity comparator");
        require(Difference(a,bad,false).bad==1,"injected corruption detection");
        a.values={-std::numeric_limits<double>::infinity()};
        require(!Difference(a,a,false).mismatch,"matching attention mask infinity");
        bad.values={std::numeric_limits<double>::infinity()};
        require(Difference(a,bad,false).mismatch,"opposite infinity detection");
        a.values={std::numeric_limits<double>::quiet_NaN()};
        require(Difference(a,a,false).mismatch,"NaN detection");
        std::cout<<"PASS: strided/F16/BF16/I32 samples, identity, injected difference, mask infinity and NaN checks\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
