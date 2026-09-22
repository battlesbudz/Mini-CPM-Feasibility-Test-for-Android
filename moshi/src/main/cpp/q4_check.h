#pragma once
#include <ggml.h>
#include <ggml-backend.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace moshi_q4 {
struct Graph {
    ggml_context* ctx = ggml_init({2*1024*1024,nullptr,true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Graph(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
    Graph()=default;
    Graph(const Graph&)=delete;
    Graph& operator=(const Graph&)=delete;
};
// Exercise the actual GGML dispatch, including batches and fused biases, before
// loading 4.3 GB of model weights. The oracle dequantizes using pinned CPU GGML
// then accumulates in double: no quantized CPU dot-product approximation.
inline uint64_t check(ggml_backend_t backend,const std::function<void(const std::string&)>& log) {
    struct Shape {int k,m,n,a2,b2,b3,bias;};
    std::vector<Shape> shapes;
    for(int cols=1;cols<=8;++cols)shapes.push_back({256,5,cols,1,1,1,cols%3});
    shapes.push_back({1024,7,1,2,4,2,2});
    shapes.push_back({4096,33,1,1,1,1,0});
    shapes.push_back({2816,9,1,1,1,1,1});
    shapes.push_back({11264,17,1,1,1,1,1});
    uint64_t checked=0;
    for(const auto& s:shapes)for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16}) {
        std::ostringstream label;
        label<<"k="<<s.k<<" rows="<<s.m<<" cols="<<s.n<<" a2="<<s.a2
             <<" b2="<<s.b2<<" b3="<<s.b3<<" bias="<<s.bias<<" input="<<ggml_type_name(type);
        log("START "+label.str());
        Graph g;if(!g.ctx)throw std::runtime_error("Q4 check context allocation failed");
        auto a=ggml_new_tensor_4d(g.ctx,GGML_TYPE_Q4_K,s.k,s.m,s.a2,s.b3);
        auto b=ggml_new_tensor_4d(g.ctx,type,s.k,s.n,s.b2,s.b3);
        auto out=ggml_mul_mat(g.ctx,a,b);
        auto bias0=ggml_dup_tensor(g.ctx,out),bias1=ggml_dup_tensor(g.ctx,out);
        auto result=out;
        if(s.bias>0)result=ggml_add(g.ctx,result,bias0);
        if(s.bias>1)result=ggml_add(g.ctx,result,bias1);
        auto graph=ggml_new_graph(g.ctx);ggml_build_forward_expand(graph,result);
        if(!ggml_backend_supports_op(backend,out))throw std::runtime_error("Q4 matrix-vector unsupported");
        g.buffer=ggml_backend_alloc_ctx_tensors(g.ctx,backend);
        if(!g.buffer)throw std::runtime_error("Q4 check buffer allocation failed");
        const size_t aCount=ggml_nelements(a),bCount=ggml_nelements(b),count=ggml_nelements(result);
        if(ggml_type_size(GGML_TYPE_Q4_K)!=144||ggml_blck_size(GGML_TYPE_Q4_K)!=256)
            throw std::runtime_error("Unexpected Q4_K layout");
        std::vector<uint32_t> packed(ggml_nbytes(a)/4);
        auto bytes=reinterpret_cast<unsigned char*>(packed.data());
        uint32_t rng=0x12345678;
        for(size_t i=0;i<ggml_nbytes(a);++i){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;bytes[i]=rng&255;}
        for(size_t block=0;block<aCount/256;++block){
            const ggml_fp16_t d=ggml_fp32_to_fp16(float(1+block%11)/1024.f);
            const ggml_fp16_t dm=ggml_fp32_to_fp16(float(1+block%7)/512.f);
            std::memcpy(bytes+block*144,&d,2);std::memcpy(bytes+block*144+2,&dm,2);
        }
        std::vector<float> af(aCount),bf(bCount),actual(count),biasData0(count),biasData1(count);
        std::vector<ggml_fp16_t> bh(bCount);
        ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(packed.data(),af.data(),aCount);
        ggml_backend_tensor_set(a,packed.data(),0,ggml_nbytes(a));
        double worst=0;
        for(int pass=0;pass<2;++pass){
            for(size_t i=0;i<bCount;++i){
                bf[i]=float(std::sin((i+pass*13)*.137)*.2);
                bh[i]=ggml_fp32_to_fp16(bf[i]);
                if(type==GGML_TYPE_F16)bf[i]=ggml_fp16_to_fp32(bh[i]);
            }
            for(size_t i=0;i<count;++i){biasData0[i]=float(int(i%11)-5)/16;biasData1[i]=float(int(i%7)-3)/32;}
            ggml_backend_tensor_set(b,type==GGML_TYPE_F16?static_cast<void*>(bh.data()):static_cast<void*>(bf.data()),0,ggml_nbytes(b));
            ggml_backend_tensor_set(bias0,biasData0.data(),0,count*4);
            ggml_backend_tensor_set(bias1,biasData1.data(),0,count*4);
            std::fill(actual.begin(),actual.end(),12345.f);
            ggml_backend_tensor_set(result,actual.data(),0,count*4);
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)
                throw std::runtime_error("Q4 dispatch failed: "+label.str());
            ggml_backend_tensor_get(result,actual.data(),0,count*4);
            double error=0,energy=0;
            for(int b3=0;b3<s.b3;++b3)for(int b2=0;b2<s.b2;++b2)
            for(int col=0;col<s.n;++col)for(int row=0;row<s.m;++row){
                size_t index=((b3*s.b2+b2)*s.n+col)*s.m+row;
                size_t ai=((b3*s.a2+b2/(s.b2/s.a2))*s.m+row)*s.k;
                size_t bi=((b3*s.b2+b2)*s.n+col)*s.k;
                double expected=0;for(int k=0;k<s.k;++k)expected+=double(af[ai+k])*bf[bi+k];
                if(s.bias>0)expected+=biasData0[index];if(s.bias>1)expected+=biasData1[index];
                double diff=actual[index]-expected;
                if(!std::isfinite(actual[index])||std::abs(diff)>0.0005+std::abs(expected)*0.00002){
                    std::ostringstream msg;msg<<"Q4 FAILED "<<label.str()<<" pass="<<pass<<" index="<<index
                        <<" expected="<<expected<<" actual="<<actual[index];throw std::runtime_error(msg.str());
                }
                error+=diff*diff;energy+=expected*expected;++checked;
            }
            double relative=std::sqrt(error/std::max(energy,1e-30));worst=std::max(worst,relative);
            if(relative>0.00002)throw std::runtime_error("Q4 relative error exceeds 0.00002: "+label.str());
            std::vector<uint32_t> after(packed.size());ggml_backend_tensor_get(a,after.data(),0,ggml_nbytes(a));
            if(after!=packed)throw std::runtime_error("Q4 weights changed during dispatch");
        }
        log("PASS "+label.str()+" nrmse="+std::to_string(worst));
    }
    log("PASS Q4_K: 24 cases / 48 dispatches, elements="+std::to_string(checked));
    return checked;
}
}
